use winit::window::Window;

use crate::graph::context::Surface;

#[derive(Debug)]
pub struct Framework<'cx, 'window> {
    context: &'cx Context,
    window: &'window Window,
    window_surface: Surface<'window>,
}

