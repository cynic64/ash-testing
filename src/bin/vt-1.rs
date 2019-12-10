use ash::extensions::khr::XlibSurface;
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::{vk, vk_make_version, Entry};

use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::ptr;

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
        api_version: vk_make_version!(1, 1, 0),
        ..Default::default()
    };

    let layer_names = [
        CString::new("VK_LAYER_LUNARG_standard_validation").unwrap(),
        CString::new("VK_LAYER_KHRONOS_validation").unwrap(),
    ];
    let layers_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let extension_names_raw = extension_names();

    let mut debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    };

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names_raw)
        .push_next(&mut debug_utils_create_info);

    let entry = Entry::new().unwrap();

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Couldn't create instance")
    };

    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);

    let debug_utils_messenger = unsafe {
        debug_utils_loader
            .create_debug_utils_messenger(&debug_utils_create_info, None)
            .expect("Debug Utils Callback")
    };

    // print info about instance
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
    dbg![entry.enumerate_instance_layer_properties().unwrap()];

    unsafe {
        instance.destroy_instance(None);
    }
}

fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}
