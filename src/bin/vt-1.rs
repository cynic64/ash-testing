use ash::extensions::khr::XlibSurface;
use ash::extensions::{
    ext::DebugReport,
    khr::Surface,
};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::{vk, vk_make_version, Entry};
use std::ffi::CString;

use winit::{Event, WindowEvent};

pub fn main() {
    // create winit window
    let mut events_loop = winit::EventsLoop::new();
    let _window = winit::WindowBuilder::new()
        .with_title("Ash - Example")
        .build(&events_loop)
        .unwrap();

    loop {
        let mut exit = false;

        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => exit = true,
            _ => {}
        });

        if exit {
            break;
        }
    }

    // create instance
    let app_info = vk::ApplicationInfo {
        api_version: vk_make_version!(1, 0, 0),
        ..Default::default()
    };

    let layer_names = [CString::new("VK_LAYER_LUNARG_standard_validation").unwrap()];
    let layers_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let extension_names_raw = extension_names();

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names_raw);

    let entry = Entry::new().unwrap();

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Couldn't create instance")
    };

    unsafe {
        instance
            .enumerate_physical_devices()
            .unwrap()
            .iter()
            .for_each(|dev| {
                dbg![instance.get_physical_device_properties(*dev)];
            });
    }

    dbg![entry.enumerate_instance_extension_properties().unwrap()];
}

fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugReport::name().as_ptr(),
    ]
}
