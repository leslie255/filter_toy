#![feature(array_chunks, f16)]
#![allow(linker_messages)]

mod graph;
mod input;
mod utils;

use std::path::PathBuf;

use graph::{
    context::{Context, Surface},
    shapes::TexturedRectangle,
    texture::Texture2d,
};

use cgmath::*;
use image::RgbaImage;
use input::InputHelper;
use utils::Wait;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

#[allow(dead_code)]
fn map_buffer(buffer_slice: wgpu::BufferSlice) {
    buffer_slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
}

/// `width * target_pixel_byte_cost` must be a multiple of `256`. This is required because `copy_texture_to_buffer` requires
/// texture with bytes per row of multiple of `256`.
/// Texture also cannot be in any compressed formats.
#[allow(dead_code)]
fn read_data_from_texture(
    context: &Context,
    width: u32,
    height: u32,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    // `debug_assert` because `block_copy_size` would return `None` later anyways.
    debug_assert!(!texture.format().is_compressed());
    let bytes_per_pixel = texture.format().block_copy_size(None).unwrap();
    assert!((width * bytes_per_pixel) % 256 == 0);
    let mut encoder = context
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let buffer_size = bytes_per_pixel * width * height;
    let staging_buffer = context.wgpu_device.create_buffer(
        &(wgpu::BufferDescriptor {
            size: buffer_size.into(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        }),
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let command_buffer = encoder.finish();
    context.wgpu_queue.submit([command_buffer]);

    let mut result_data = Vec::<u8>::with_capacity(buffer_size as usize);
    let buffer_slice = staging_buffer.slice(..);
    map_buffer(buffer_slice);
    context.wgpu_device.poll(wgpu::MaintainBase::Wait);
    let view = buffer_slice.get_mapped_range();
    result_data.extend_from_slice(&view[..]);
    drop(view);
    staging_buffer.unmap();

    result_data
}

/// Manages scrolling and scaling of a canvas.
#[derive(Debug, Clone, Copy)]
struct Canvas {
    offset: Vector2<f32>,
    scale: f32,
    scale_factor: f32,
}

#[allow(dead_code)]
impl Canvas {
    pub fn new(scale_factor: f32) -> Self {
        Self {
            offset: vec2(0., 0.),
            scale: 1.,
            scale_factor,
        }
    }

    pub fn physical_scale(&self) -> f32 {
        self.scale * self.scale_factor
    }

    pub fn model(&self, position: Point2<f32>) -> Matrix4<f32> {
        let translation = self.offset + position.to_vec();
        Matrix4::from_scale(self.physical_scale())
            * Matrix4::from_translation(translation.extend(0.))
    }

    pub fn inverse_model(&self) -> Matrix4<f32> {
        self.model(point2(0., 0.))
            .invert()
            .unwrap_or(Matrix4::identity())
    }

    /// Transform a point from screen-space back to in-canvas world space.
    /// `point` must be coordinate with `(0, 0)` at center of screen.
    pub fn inverse_transform(&self, point: Point2<f32>) -> Point2<f32> {
        let point3 = point3(point.x, point.y, 0.);
        let point_world = self.inverse_model().transform_point(point3);
        point2(point_world.x, point_world.y)
    }

    pub fn move_(&mut self, delta: Vector2<f32>) {
        self.offset += delta / self.physical_scale();
    }

    pub fn scale(&mut self, delta: f32, position_centered: Point2<f32>) {
        let mut scale = self.scale.ln();
        scale += delta;
        if delta.is_sign_positive() {
            self.move_(-position_centered.to_vec() * delta);
        }
        self.scale = scale.exp();
        if delta.is_sign_negative() {
            self.move_(-position_centered.to_vec() * delta);
        }
    }
}

struct Application<'cx, 'window> {
    window: Option<&'window Window>,
    window_surface: Surface<'window>,
    context: &'cx Context,
    canvas: Canvas,
    input_helper: InputHelper,
    /// If the current frame was the first frame after a resize.
    is_first_frame_after_resize: bool,
    frame_counter: u64,
    image_updated: bool,
    image: Option<RgbaImage>,
    image_path: Option<PathBuf>,
    image_rectangle: Option<TexturedRectangle<'cx>>,
}

impl<'cx, 'window> Application<'cx, 'window> {
    pub fn new(
        window: Option<&'window Window>,
        wgpu_surface: wgpu::Surface<'window>,
        context: &'cx Context,
    ) -> Self {
        let image_path = std::env::args().nth(1).map(PathBuf::from);
        let image = image_path
            .as_deref()
            .map(|path| image::open(path).unwrap().into_rgba8());
        Self {
            window,
            window_surface: Surface::for_window(
                window,
                wgpu_surface,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ),
            context,
            canvas: Canvas::new(window.map_or(0.0, |w| w.scale_factor() as f32)),
            input_helper: InputHelper::new(),
            is_first_frame_after_resize: false,
            frame_counter: 0,
            image_updated: image_path.is_some(),
            image,
            image_path,
            image_rectangle: None,
        }
    }

    fn draw(&mut self) {
        self.window_surface.begin_drawing();

        let mut encoder = self
            .context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut render_pass = self.window_surface.create_render_pass(
            &mut encoder,
            Some(wgpu::Color::BLACK),
            None,
            None,
            None,
        );

        if self.image_updated {
            self.image_updated = false;
            self.update_texture();
        }
        if let Some(ref mut image_rectangle) = self.image_rectangle {
            let position = point2(0.0, 0.0) - image_rectangle.size() / 2.0;
            image_rectangle.draw(
                &self.window_surface,
                &mut render_pass,
                self.canvas.model(position),
            );
        }

        drop(render_pass);

        self.context.wgpu_queue.submit([encoder.finish()]);

        if let Some(window) = self.window {
            window.pre_present_notify();
        }

        self.window_surface.present();

        self.is_first_frame_after_resize = false;
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    fn update_texture(&mut self) {
        let Some(ref image) = self.image else {
            self.image_rectangle = None;
            return;
        };
        let image_size = vec2(image.width(), image.height());
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = Texture2d::new(self.context, image_size, format);
        texture.write(self.context, image);
        let rectangle =
            TexturedRectangle::new(self.context, &self.window_surface, texture.create_view())
                .with_size(image_size.map(|u| u as f32));
        self.image_rectangle = Some(rectangle);
    }

    fn window_scale_factor(&self) -> f64 {
        self.window.map_or(0.0, |window| window.scale_factor())
    }

    fn paint(&mut self) {
        let position = self
            .input_helper
            .cursor_position_physical()
            .unwrap_or(point2(0., 0.));
        let center = position - self.window_surface.size() / 2.;
        let center = point2(center.x, -center.y);
        let world_coord = self.canvas.inverse_transform(center);
        if let Some(ref mut image) = self.image {
            let image_coord_f = vec2(world_coord.x, -world_coord.y)
                + vec2(image.width() as f32, image.height() as f32) / 2.0;
            let image_coord = image_coord_f.map(|f| f as u32);
            *image.get_pixel_mut(image_coord.x, image_coord.y) = [255, 0, 255, 255].into();
            self.image_updated = true;
            self.draw();
        }
    }
}

impl ApplicationHandler for Application<'_, '_> {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.draw();
            }
            WindowEvent::Resized(physical_size) => {
                self.is_first_frame_after_resize = true;
                let size = vec2(physical_size.width as f32, physical_size.height as f32);
                self.window_surface.window_resized(self.context, size);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer: _,
            } => {
                self.is_first_frame_after_resize = true;
                if self.frame_counter != 0 {
                    self.draw();
                }
                self.canvas.scale_factor = scale_factor as f32;
            }
            WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.input_helper
                    .notify_cursor_moved(position, self.window_scale_factor());
                if self.input_helper.button_is_pressed(0) {
                    self.paint();
                }
            }
            WindowEvent::CursorEntered { device_id: _ } => {
                self.input_helper.notify_cursor_entered();
            }
            WindowEvent::CursorLeft { device_id: _ } => {
                self.input_helper.notify_cursor_left();
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.input_helper.notify_key_event(&event);
                'block: {
                    if !event.state.is_pressed() {
                        break 'block;
                    }
                    let key_code = match event.physical_key {
                        winit::keyboard::PhysicalKey::Code(key_code) => key_code,
                        _ => break 'block,
                    };
                    match key_code {
                        winit::keyboard::KeyCode::ArrowUp => self.canvas.move_(vec2(0.0, -12.0)),
                        winit::keyboard::KeyCode::ArrowDown => self.canvas.move_(vec2(0.0, 12.0)),
                        winit::keyboard::KeyCode::ArrowRight => self.canvas.move_(vec2(-12.0, 0.0)),
                        winit::keyboard::KeyCode::ArrowLeft => self.canvas.move_(vec2(12.0, 0.0)),
                        winit::keyboard::KeyCode::Equal | winit::keyboard::KeyCode::NumpadAdd
                            if self.input_helper.super_is_down()
                                | self.input_helper.control_is_down() =>
                        {
                            self.canvas.scale(1.0 / 12.0, point2(0.0, 0.0))
                        }
                        winit::keyboard::KeyCode::Minus
                        | winit::keyboard::KeyCode::NumpadSubtract
                            if self.input_helper.super_is_down()
                                | self.input_helper.control_is_down() =>
                        {
                            self.canvas.scale(-1.0 / 12.0, point2(0.0, 0.0))
                        }
                        _ => (),
                    }
                    self.draw();
                }
            }
            WindowEvent::MouseWheel {
                device_id: _,
                delta,
                phase: _,
            } => {
                let delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => vec2(x, y) * 12.0,
                    winit::event::MouseScrollDelta::PixelDelta(delta) => {
                        vec2(delta.x, delta.y).map(|f| f as f32)
                    }
                };
                if self
                    .input_helper
                    .key_is_down(winit::keyboard::KeyCode::AltLeft)
                {
                    let position = self
                        .input_helper
                        .cursor_position_physical()
                        .unwrap_or(point2(0., 0.));
                    let center = position - self.window_surface.size() / 2.;
                    let center = point2(center.x, -center.y);
                    self.canvas.scale(delta.y / 100.0, center);
                } else {
                    self.canvas.move_(vec2(delta.x, -delta.y));
                }
                self.draw();
            }
            WindowEvent::PinchGesture {
                device_id: _,
                delta,
                phase: _,
            } => {
                let position = self
                    .input_helper
                    .cursor_position_physical()
                    .unwrap_or(point2(0., 0.));
                let center = position - self.window_surface.size() / 2.;
                let center = point2(center.x, -center.y);
                self.canvas.scale(delta as f32, center);
                self.draw();
            }
            WindowEvent::DroppedFile(path) => {
                self.image_updated = true;
                if let Ok(image) = time!(image::open(&path)) {
                    self.image = Some(image.into_rgba8());
                    self.image_path = Some(path);
                    self.draw();
                } else {
                    println!("[INFO] Unable to open image file at path {path:?}");
                    self.image = None;
                    self.image_path = None;
                    self.draw();
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        #[allow(clippy::single_match)]
        match event {
            winit::event::DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.input_helper.button_is_pressed(2) {
                    self.canvas
                        .move_((vec2(dx, -dy) * self.window_scale_factor()).map(|f| f as f32));
                    self.draw();
                }
            }
            winit::event::DeviceEvent::Button { button, state } => {
                self.input_helper.notify_button_event(button, state);
                if button == 0 && state.is_pressed() {
                    self.paint();
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let event_loop = EventLoop::new().unwrap();
    #[allow(deprecated)]
    let window = event_loop
        .create_window(Window::default_attributes().with_title("Filter Toy"))
        .unwrap();
    let (wgpu_surface, context) = Context::for_window(wgpu_instance, &window).wait();
    let mut application = Application::new(Some(&window), wgpu_surface, &context);
    event_loop.run_app(&mut application).unwrap();
}
