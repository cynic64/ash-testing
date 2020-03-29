use ash::extensions::khr::{Swapchain, XlibSurface};
use ash::extensions::{ext::DebugUtils, khr::Surface};
use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk::DeviceSize;
use ash::{vk, vk_make_version, Entry};

use winit::dpi::LogicalPosition;
use winit::{ElementState, Event, MouseButton, WindowEvent};

use crossbeam_channel::{Receiver, Sender};

use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;

use memoffset::offset_of;

use ash_testing::{get_elapsed, relative_path, size_of_slice, LoopTimer};

const MAX_FRAMES_IN_FLIGHT: usize = 4;

#[cfg(target_os = "macos")]
extern crate cocoa;
#[cfg(target_os = "macos")]
extern crate metal;
#[cfg(target_os = "macos")]
extern crate objc;
extern crate winit;
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;
#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;
#[cfg(target_os = "macos")]
use std::mem;

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;

type IndexType = u32;

// 0-capacity channels have the best performance
const MESH_CHANNEL_CAPACITY: usize = 0;

const KINKINESS: f64 = 0.2;

// range from 0.0 .. screen size
type PixelPos = [f64; 2];

// -1.0 .. 1.0
type VkPos = [f64; 2];

// we cannot know which image will be used before calling acquire_next_image,
// for which we already need a semaphore ready for it to signal.

// therefore, there is no direct link between FlyingFrames and swapchain images
// - each time we draw a new frame, wait for the oldest sync_set to finish and
// use that.
struct FlyingFrame {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    render_finished_fence: vk::Fence,
    command_buffer: Option<vk::CommandBuffer>,

    vertex_staging_buffer: vk::Buffer,
    vertex_staging_buffer_memory: vk::DeviceMemory,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_staging_buffer: vk::Buffer,
    index_staging_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    vbuf_copied_fence: vk::Fence,
    ibuf_copied_fence: vk::Fence,
}

#[repr(C)]
#[derive(Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<IndexType>,
}

pub fn main() {
    // create winit window
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title("Ash - Example")
        .build(&events_loop)
        .unwrap();
    let hidpi_factor = window.get_hidpi_factor();

    // create instance
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: CString::new("Thingy").unwrap().as_ptr(),
        application_version: 0,
        p_engine_name: CString::new("Vulkan Engine").unwrap().as_ptr(),
        engine_version: 0,
        api_version: vk_make_version!(1, 1, 0),
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

    let debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
            | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
            // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(vulkan_debug_utils_callback),
        p_user_data: ptr::null_mut(),
    };

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        // setting this to a null pointer also works, but leave it like this to
        // be safe i guess?
        p_next: &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
            as *const c_void,
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        enabled_layer_count: layer_names.len() as u32,
        pp_enabled_layer_names: layers_names_raw.as_ptr(),
        enabled_extension_count: extension_names_raw.len() as u32,
        pp_enabled_extension_names: extension_names_raw.as_ptr(),
    };

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

    // create surface
    let surface =
        unsafe { create_surface(&entry, &instance, &window) }.expect("couldn't create surface");

    let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);

    // get physical device
    let physical_device = {
        let phys_devs = unsafe { instance.enumerate_physical_devices() }
            .expect("Couldn't enumerate physical devices");

        *phys_devs
            .iter()
            .find(|phys_dev| is_phys_dev_suitable(&instance, phys_dev))
            .expect("No suitable physical device found!")
    };

    // get queue family index
    let queue_family_index: u32 = {
        let queues =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        queues
            .iter()
            .enumerate()
            .find(|(_idx, queue)| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .expect("Couldn't find a graphics queue")
            .0
            .try_into()
            .unwrap()
    };

    if !unsafe {
        surface_loader.get_physical_device_surface_support(
            physical_device,
            queue_family_index,
            surface,
        )
    } {
        panic!("Queue does not have surface support! It's possible that a separate queue with surface support exists, but the current implementation is not capable of finding one.");
    }

    // get logical device
    let device_queue_create_info = vk::DeviceQueueCreateInfo {
        s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceQueueCreateFlags::empty(),
        queue_family_index,
        queue_count: 1,
        p_queue_priorities: [1.0].as_ptr(),
    };

    let device_extensions_raw = get_device_extensions_raw();

    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: 1,
        p_queue_create_infos: [device_queue_create_info].as_ptr(),
        // not used by Vulkan anymore
        enabled_layer_count: 0,
        pp_enabled_layer_names: ptr::null(),
        // these are
        enabled_extension_count: device_extensions_raw.len() as u32,
        pp_enabled_extension_names: device_extensions_raw.as_ptr(),
        p_enabled_features: &vk::PhysicalDeviceFeatures::builder().build(),
    };

    let device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Couldn't create device")
    };

    // get queue (0 = take first queue)
    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    let swapchain_creator = Swapchain::new(&instance, &device);

    let (mut swapchain, mut swapchain_images, mut swapchain_dims) = create_swapchain(
        physical_device,
        &surface_loader,
        surface,
        &swapchain_creator,
    );

    println!("Swapchain image count: {}", swapchain_images.len());
    println!("Maximum frames in flight: {} ", MAX_FRAMES_IN_FLIGHT);

    let mut swapchain_image_views = create_swapchain_image_views(&device, &swapchain_images);

    // shaders
    let frag_code = read_shader_code(&relative_path("shaders/vt-5-vbuf/triangle.frag.spv"));
    let vert_code = read_shader_code(&relative_path("shaders/vt-5-vbuf/triangle.vert.spv"));

    let frag_module = create_shader_module(&device, frag_code);
    let vert_module = create_shader_module(&device, vert_code);

    let entry_point = CString::new("main").unwrap();

    let vert_stage_info = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineShaderStageCreateFlags::empty(),
        stage: vk::ShaderStageFlags::VERTEX,
        module: vert_module,
        p_name: entry_point.as_ptr(),
        p_specialization_info: &vk::SpecializationInfo::default(),
    };

    let frag_stage_info = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineShaderStageCreateFlags::empty(),
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: frag_module,
        p_name: entry_point.as_ptr(),
        p_specialization_info: &vk::SpecializationInfo::default(),
    };

    let shader_stages = [vert_stage_info, frag_stage_info];

    // command pool
    let command_pool = create_command_pool(&device, queue_family_index);

    // mesh allocation sizes
    let est_vertices = 100_000;
    let est_indices = 200_000;
    let vertex_buffer_capacity = (est_vertices * std::mem::size_of::<Vertex>()) as DeviceSize;
    let index_buffer_capacity = (est_indices * std::mem::size_of::<IndexType>()) as DeviceSize;

    // render pass
    let render_pass = create_render_pass(&device);

    // pipeline layout
    let pipeline_layout = create_pipeline_layout(&device);

    // pipeline
    let binding_descriptions = Vertex::get_binding_descriptions();
    let attribute_descriptions = Vertex::get_attribute_descriptions();

    let mut pipeline = create_pipeline(
        &device,
        render_pass,
        pipeline_layout,
        swapchain_dims,
        shader_stages,
        &binding_descriptions,
        &attribute_descriptions,
    );

    // framebuffer creation
    let mut framebuffers =
        create_framebuffers(&device, render_pass, swapchain_dims, &swapchain_image_views);

    // flying frames (one for each swapchain image)
    let device_memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let mut flying_frames = setup_flying_frames(
        &device,
        device_memory_properties,
        vertex_buffer_capacity,
        index_buffer_capacity,
    );

    // used to check whether rendering to a swapchain image has finished,
    // necessary because we have more sync sets than swapchain images
    let mut swapchain_fences: Vec<Option<vk::Fence>> = vec![None; swapchain_images.len()];

    // used to calculate FPS and keep track of which sync set to use next
    // (remember, it's independent from which swapchain image is being used)
    let mut frames_drawn = 0;
    let start_time = std::time::Instant::now();

    let mut must_recreate_swapchain = false;

    // start the mesh generating thread (will constantly create new mesh data
    // and send it to the rendering thread across a channel)

    // the spawned thread will send us new meshes
    let (mesh_send, mesh_recv): (Sender<Mesh>, Receiver<Mesh>) =
        crossbeam_channel::bounded(MESH_CHANNEL_CAPACITY);
    // and we will send it mouse click locations (unbounded because rendering is
    // usually faster and multiple clicks can be processed by the meshing thread
    // in one iteration)
    let (click_send, click_recv): (Sender<VkPos>, Receiver<VkPos>) = crossbeam_channel::unbounded();
    // spawn it
    let mesh_gen_handle = std::thread::spawn(move || mesh_thread(mesh_send, click_recv));

    // winit does not have a function to get the current mouse position, so
    // instead we keep track of it and update it every time the mouse moves
    let mut mouse_pos = [0.0, 0.0];

    // also track whether the LMB is down or not
    let mut mouse_down = false;

    // and last location we sent to the mesh-generating thread (because sending
    // locations very close together is pointless)
    let mut last_sent_mouse_pos: PixelPos = [-9999.0, -9999.0];
    // at least 10 pixels in between every sending of the mouse position
    let mouse_pos_send_thresh_dist = 10.0;

    // timers
    let mut timer_mesh_wait = LoopTimer::new("Waiting on mesh gen".to_string());
    let mut timer_draw = LoopTimer::new("Drawing".to_string());
    let mut timer_write_vbuf = LoopTimer::new("Write vbuf".to_string());
    let mut timer_copy_vbuf = LoopTimer::new("Copy vbuf".to_string());
    let mut timer_write_ibuf = LoopTimer::new("Write ibuf".to_string());
    let mut timer_copy_ibuf = LoopTimer::new("Copy ibuf".to_string());
    let mut timer_copy_complete = LoopTimer::new("Copy completion".to_string());

    loop {
        if must_recreate_swapchain {
            unsafe { device.device_wait_idle() }.expect("Couldn't wait for device to become idle");

            // Cleanup_swapchain requires ownership of these to destroy
            // them. They will be re-created later. Technically I think it
            // would also work to pass pointers to cleanup_swapchain instead
            // of transferring ownership, but I don't like the idea of that
            // because then values in command_buffers and framebuffers would
            // point to destroyed vulkan objects.
            let our_framebuffers = std::mem::replace(&mut framebuffers, vec![]);
            let our_swapchain_image_views = std::mem::replace(&mut swapchain_image_views, vec![]);

            cleanup_swapchain(
                &device,
                &swapchain_creator,
                our_framebuffers,
                pipeline,
                our_swapchain_image_views,
                swapchain,
            );

            // re-create swapchain
            let ret = create_swapchain(
                physical_device,
                &surface_loader,
                surface,
                &swapchain_creator,
            );

            swapchain = ret.0;
            swapchain_images = ret.1;
            swapchain_dims = ret.2;

            // re-create image views
            swapchain_image_views = create_swapchain_image_views(&device, &swapchain_images);

            // re-create graphics pipeline
            pipeline = create_pipeline(
                &device,
                render_pass,
                pipeline_layout,
                swapchain_dims,
                shader_stages,
                &binding_descriptions,
                &attribute_descriptions,
            );

            // re-create framebuffers
            // framebuffer creation
            framebuffers =
                create_framebuffers(&device, render_pass, swapchain_dims, &swapchain_image_views);

            must_recreate_swapchain = false;
        }

        let mut exit = false;

        events_loop.poll_events(|ev| match ev {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => exit = true,
            Event::WindowEvent {
                event:
                    WindowEvent::CursorMoved {
                        position: LogicalPosition { x, y },
                        ..
                    },
                ..
            } => mouse_pos = [x * hidpi_factor, y * hidpi_factor],
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => match state {
                ElementState::Pressed => mouse_down = true,
                ElementState::Released => mouse_down = false,
            },
            _ => {}
        });

        if exit {
            break;
        }

        if mouse_down {
            // only send mouse pos if it's over a certain distance from the
            // previous position sent
            if pixel_dist(&mouse_pos, &last_sent_mouse_pos) > mouse_pos_send_thresh_dist {
                click_send
                    .send(pixel_to_vk(swapchain_dims, &mouse_pos))
                    .unwrap();
                last_sent_mouse_pos = mouse_pos;
            }
        }

        let flying_frame_idx = frames_drawn % MAX_FRAMES_IN_FLIGHT;
        let frame = &flying_frames[flying_frame_idx];

        let image_available_semaphore = frame.image_available_semaphore;
        let render_finished_semaphore = frame.render_finished_semaphore;
        let render_finished_fence = frame.render_finished_fence;

        // we can't use this sync set until whichever rendering was using it
        // previously is finished, so wait for rendering to finished (use the
        // fence because it's a GPU - CPU sync)
        unsafe { device.wait_for_fences(&[render_finished_fence], true, std::u64::MAX) }
            .expect("Couldn't wait for previous sync set to finish rendering");

        // since rendering has finished, we can free that command buffer
        if let Some(command_buffer) = frame.command_buffer {
            unsafe { device.free_command_buffers(command_pool, &[command_buffer]) };
        } else {
            // should only happen when swapchain images are first used, panic
            // otherwise
            if frames_drawn >= MAX_FRAMES_IN_FLIGHT {
                panic!(
                    "No command buffer in flying frame on frame {}",
                    frames_drawn
                );
            }
        }

        // image_available_semaphore will be signalled once the swapchain image
        // is actually available and not being displayed anymore -
        // acquire_next_image will return the instant it knows which image index
        // will be free next, so we need to wait on that semaphore
        let acquire_result = unsafe {
            swapchain_creator.acquire_next_image(
                swapchain,
                std::u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };

        let image_idx = match acquire_result {
            Ok((image_idx, _is_sub_optimal)) => image_idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                must_recreate_swapchain = true;
                continue;
            }
            Err(e) => panic!("Unexpected error during acquire_next_image: {}", e),
        };

        // because we might have more sync sets than swapchain images, it's
        // possible another set of sync sets is still rendering to this
        // swapchain image. by waiting on the fence associated with this
        // swapchain image, we can ensure it really is available
        if let Some(image_fence) = swapchain_fences[image_idx as usize] {
            unsafe { device.wait_for_fences(&[image_fence], true, std::u64::MAX) }
                .expect("Couldn't wait for image_in_flight fence");
        } else {
            // this should only happen on the first MAX_FRAMES_IN_FLIGHT frames
            // drawn, because at that point we will not yet have started using
            // all sync sets

            // if it happens afterwards, panic
            if frames_drawn >= MAX_FRAMES_IN_FLIGHT {
                panic!(
                    "No fence for this swapchain image at frame {}! image_idx: {}",
                    frames_drawn, image_idx
                );
            }
        }

        timer_draw.stop();

        timer_mesh_wait.start();
        let mesh = mesh_recv.recv().expect("Error when receiving mesh");
        timer_mesh_wait.stop();

        // set the render_finished_fence associated with this swapchain image to
        // the fence that will be signalled when we finish rendering - in other
        // words, "We're using this image! Don't touch it till we finish."
        swapchain_fences[image_idx as usize] = Some(render_finished_fence);

        // change mesh data on GPU

        // make sure we won't go past the memory we've allocated
        let vertex_buffer_size = size_of_slice(&mesh.vertices);
        let index_buffer_size = size_of_slice(&mesh.indices);

        assert!(
            vertex_buffer_size < vertex_buffer_capacity,
            "Vertex buffers ({}) too small for vertex data ({})!",
            vertex_buffer_capacity,
            vertex_buffer_size
        );

        assert!(
            index_buffer_size < index_buffer_capacity,
            "Index buffers ({}) too small for index data ({})!",
            index_buffer_capacity,
            index_buffer_size
        );

        // even though we modify these, we don't need to mutably borrow because
        // they are magical Vulkan pointers that don't care about lifetimes
        let vertex_staging_buffer = flying_frames[flying_frame_idx].vertex_staging_buffer;
        let vertex_staging_buffer_memory =
            flying_frames[flying_frame_idx].vertex_staging_buffer_memory;
        let vertex_buffer = flying_frames[flying_frame_idx].vertex_buffer;
        let index_buffer = flying_frames[flying_frame_idx].index_buffer;
        let index_staging_buffer = flying_frames[flying_frame_idx].index_staging_buffer;
        let index_staging_buffer_memory =
            flying_frames[flying_frame_idx].index_staging_buffer_memory;

        let vbuf_copied_fence = flying_frames[flying_frame_idx].vbuf_copied_fence;
        let ibuf_copied_fence = flying_frames[flying_frame_idx].ibuf_copied_fence;

        // map and write vertex data to vertex staging buffer
        let mut copy_command_buffers = None;
        if mesh.vertices.len() > 0 {
            // reset the fences that will be used
            unsafe { device.reset_fences(&[vbuf_copied_fence, ibuf_copied_fence]) }
                .expect("Couldn't reset vbuf_ and ibuf_copied_fence");

            timer_write_vbuf.start();
            write_to_cpu_accessible_buffer(&device, vertex_staging_buffer_memory, &mesh.vertices);
            timer_write_vbuf.stop();

            // copy staging buffer to vertex buffer
            timer_copy_vbuf.start();
            let vbuf_command_buffer = copy_buffer(
                &device,
                queue,
                command_pool,
                vertex_staging_buffer,
                vertex_buffer,
                vertex_buffer_size,
                vbuf_copied_fence,
            );
            timer_copy_vbuf.stop();

            // map and write index data to staging buffer
            timer_write_ibuf.start();
            write_to_cpu_accessible_buffer(&device, index_staging_buffer_memory, &mesh.indices);
            timer_write_ibuf.stop();

            // copy staging buffer to index buffer
            timer_copy_ibuf.start();
            let ibuf_command_buffer = copy_buffer(
                &device,
                queue,
                command_pool,
                index_staging_buffer,
                index_buffer,
                index_buffer_size,
                ibuf_copied_fence,
            );
            timer_copy_ibuf.stop();

            copy_command_buffers = Some((vbuf_command_buffer, ibuf_command_buffer));
        }

        // create command buffer
        let command_buffer = create_command_buffer(
            &device,
            render_pass,
            pipeline,
            command_pool,
            framebuffers[image_idx as usize],
            swapchain_dims,
            vertex_buffer,
            index_buffer,
            mesh.indices.len() as u32,
        );

        flying_frames[frames_drawn % MAX_FRAMES_IN_FLIGHT].command_buffer = Some(command_buffer);

        if let Some((vbuf_cbuf, ibuf_cbuf)) = copy_command_buffers {
            timer_copy_complete.start();
            // before we submit, wait for both copy operations to finish
            unsafe {
                device.wait_for_fences(&[vbuf_copied_fence, ibuf_copied_fence], true, std::u64::MAX)
            }
            .expect("Couldn't wait for vertex and index buffer to finish copying");
            timer_copy_complete.stop();

            // free the command buffers used to copy
            unsafe { device.free_command_buffers(command_pool, &[vbuf_cbuf, ibuf_cbuf]) };
        }

        // submit command buffer
        let wait_semaphores = [image_available_semaphore];

        // "Each entry in the waitStages array corresponds to the semaphore with
        // the same index in pWaitSemaphores."
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

        let cur_command_buffers = [command_buffer];

        let signal_semaphores = [render_finished_semaphore];

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: cur_command_buffers.as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        };

        timer_draw.start();

        // somebody else was previously using this fence and we waited until it
        // was signalled (operation completed). now we need to reset it, because
        // we aren't yet done but the fence says we are.
        unsafe { device.reset_fences(&[render_finished_fence]) }
            .expect("Couldn't reset render_finished_fence");

        let submissions = [submit_info];
        unsafe { device.queue_submit(queue, &submissions, render_finished_fence) }
            .expect("Couldn't submit command buffer");

        // present result to swapchain
        let swapchains = [swapchain];
        let image_indices = [image_idx];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            p_results: ptr::null_mut(),
        };

        match unsafe { swapchain_creator.queue_present(queue, &present_info) } {
            Ok(_idk_what_this_is) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                must_recreate_swapchain = true;
                continue;
            }
            Err(e) => panic!("Unexpected error during queue_present: {}", e),
        };

        frames_drawn += 1;
    }

    drop(mesh_recv);

    println!("FPS: {:.2}", frames_drawn as f64 / get_elapsed(start_time));
    println!(
        "Average delta in ms: {:.5}",
        get_elapsed(start_time) / frames_drawn as f64 * 1_000.0
    );

    timer_draw.print();
    timer_mesh_wait.print();
    timer_write_vbuf.print();
    timer_copy_vbuf.print();
    timer_write_ibuf.print();
    timer_copy_ibuf.print();
    timer_copy_complete.print();

    unsafe { device.device_wait_idle() }.expect("Couldn't wait for device to become idle");

    // destroy objects
    unsafe {
        flying_frames.iter().for_each(|f| {
            if let Some(command_buffer) = f.command_buffer {
                device.free_command_buffers(command_pool, &[command_buffer]);
            }

            device.destroy_semaphore(f.image_available_semaphore, None);
            device.destroy_semaphore(f.render_finished_semaphore, None);
            device.destroy_fence(f.render_finished_fence, None);

            device.destroy_buffer(f.vertex_staging_buffer, None);
            device.destroy_buffer(f.vertex_buffer, None);
            device.destroy_buffer(f.index_staging_buffer, None);
            device.destroy_buffer(f.index_buffer, None);

            device.free_memory(f.vertex_staging_buffer_memory, None);
            device.free_memory(f.vertex_buffer_memory, None);
            device.free_memory(f.index_staging_buffer_memory, None);
            device.free_memory(f.index_buffer_memory, None);

            device.destroy_fence(f.vbuf_copied_fence, None);
            device.destroy_fence(f.ibuf_copied_fence, None);
        });

        device.destroy_shader_module(frag_module, None);
        device.destroy_shader_module(vert_module, None);

        cleanup_swapchain(
            &device,
            &swapchain_creator,
            framebuffers,
            pipeline,
            swapchain_image_views,
            swapchain,
        );

        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_render_pass(render_pass, None);

        device.destroy_command_pool(command_pool, None);

        device.destroy_device(None);

        surface_loader.destroy_surface(surface, None);

        debug_utils_loader.destroy_debug_utils_messenger(debug_utils_messenger, None);

        instance.destroy_instance(None);
    }

    println!("waiting on mesh thread");
    mesh_gen_handle.join().unwrap();
}

fn mesh_thread(mesh_send: Sender<Mesh>, click_recv: Receiver<VkPos>) {
    let mut timer = LoopTimer::new("Mesh generation".to_string());

    let mut points = vec![];

    loop {
        timer.start();

        click_recv.try_iter().for_each(|pos| points.push(pos));

        let mesh = create_mesh(&points);

        timer.stop();

        match mesh_send.send(mesh) {
            Ok(()) => {}
            Err(_) => {
                println!("Mesh receiver disconnected, mesh gen thread quitting");
                timer.print();
                return;
            }
        }
    }
}

fn create_mesh(points: &[VkPos]) -> Mesh {
    use lyon::math::Point;
    use lyon::path::Path;
    use lyon::tessellation::*;
    use lyon_tessellation::{LineCap, StrokeOptions, StrokeTessellator};

    let mut builder = Path::builder();

    // need at least 2 points
    if points.len() < 2 {
        return Mesh {
            vertices: vec![],
            indices: vec![],
        };
    }

    // linearly extrapolate one point on either end of the point list to so we
    // can actually draw the first and last point, rather than using them as
    // guides for control points
    // extrapolate backwards: a - (b - a) = 2a - b
    let first_point = vec![[
        2.0 * points[0][0] - points[1][0],
        2.0 * points[0][1] - points[1][1],
    ]];
    // extrapolate forwards: b + (b - a) = 2b - a
    let len = points.len();
    let last_point = vec![[
        2.0 * points[len - 1][0] - points[len - 2][0],
        2.0 * points[len - 1][1] - points[len - 2][1],
    ]];

    let points: Vec<_> = first_point
        .iter()
        .chain(points)
        .chain(&last_point)
        .collect();

    builder.move_to(vk_to_point(&points[1]));

    // last point is used as a control point, and penultimate point is joined
    // to, so only go to len - 2
    for i in 1..points.len() - 2 {
        // we join x and y, using w and z to dictate the control points (a and b)
        let x = points[i];
        let y = points[i + 1];
        let w = points[i - 1];
        let z = points[i + 2];

        let a = [
            x[0] + KINKINESS * y[0] - KINKINESS * w[0],
            x[1] + KINKINESS * y[1] - KINKINESS * w[1],
        ];
        let b = [
            y[0] - KINKINESS * z[0] + KINKINESS * x[0],
            y[1] - KINKINESS * z[1] + KINKINESS * x[1],
        ];

        builder.cubic_bezier_to(vk_to_point(&a), vk_to_point(&b), vk_to_point(&y));
    }

    let path = builder.build();

    // will contain the result of the tessellation.
    let mut geometry: VertexBuffers<Vertex, IndexType> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();
    {
        // compute the tessellation.
        tessellator
            .tessellate(
                &path,
                &StrokeOptions::DEFAULT
                    .with_line_width(0.01)
                    .with_tolerance(0.000001)
                    .with_start_cap(LineCap::Round)
                    .with_end_cap(LineCap::Round),
                &mut BuffersBuilder::new(&mut geometry, |pos: Point, _: StrokeAttributes| Vertex {
                    position: pos.to_array(),
                    color: [
                        pos.x * 0.5 + 0.5,
                        pos.y * 0.5 + 0.5,
                        (pos.x * 0.5 + 0.5) * (pos.y * 0.5 + 0.5),
                    ],
                }),
            )
            .unwrap();
    }

    Mesh {
        vertices: geometry.vertices,
        indices: geometry.indices,
    }
}

fn write_to_cpu_accessible_buffer<D: DeviceV1_0, T>(
    device: &D,
    buffer_memory: vk::DeviceMemory,
    data: &[T],
) {
    let buffer_size = size_of_slice(data);

    unsafe {
        let mapped_memory = device
            .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            .expect("Couldn't map vertex buffer memory") as *mut T;

        mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());

        device.unmap_memory(buffer_memory);
    }
}

fn copy_buffer<D: DeviceV1_0>(
    device: &D,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    source: vk::Buffer,
    dest: vk::Buffer,
    bytes: DeviceSize,
    fence: vk::Fence,
) -> vk::CommandBuffer {
    // returns the command buffer used to perform the copy operation, which
    // should be freed after the given fence is signalled

    // fence will be signalled when the copy operation finishes
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        // Primary: can be submitted to a queue, but not called from other command buffers
        // Secondary: can't be directly submitted, but can be called from other command buffers
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };

    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
        .expect("Couldn't allocate command buffers")[0];

    // begin command buffer
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        p_inheritance_info: ptr::null(),
    };

    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
        .expect("Couldn't begin command buffer");

    // copy
    let copy_region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: bytes,
    }];

    unsafe { device.cmd_copy_buffer(command_buffer, source, dest, &copy_region) };

    unsafe { device.end_command_buffer(command_buffer) }.expect("Couldn't record command buffer!");

    let command_buffers = [command_buffer];

    // submit
    let submit_info = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: command_buffers.as_ptr(),
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
    }];

    unsafe { device.queue_submit(queue, &submit_info, fence) }
        .expect("Couldn't submit command buffer");

    command_buffer
}

fn create_buffer<D: DeviceV1_0>(
    device: &D,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    bytes: DeviceSize,
    usage: vk::BufferUsageFlags,
    required_memory_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let vertex_buffer_info = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::BufferCreateFlags::empty(),
        size: bytes,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,

        // don't understand why this can be left blank
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
    };

    let vertex_buffer = unsafe { device.create_buffer(&vertex_buffer_info, None) }
        .expect("Couldn't create vertex buffer");

    let vertex_buffer_memory_requirements =
        unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

    let vertex_buffer_memory_type_index = find_memory_type(
        device_memory_properties,
        vertex_buffer_memory_requirements.memory_type_bits,
        required_memory_properties,
    );

    let vertex_buffer_alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),

        // this size can be different from the size in vertex_buffer_info! I
        // think the minimum allocation on the GPU is 256 bytes or something.
        allocation_size: vertex_buffer_memory_requirements.size,
        memory_type_index: vertex_buffer_memory_type_index,
    };

    let vertex_buffer_memory = unsafe { device.allocate_memory(&vertex_buffer_alloc_info, None) }
        .expect("Couldn't allocate vertex buffer device memory");

    unsafe { device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0) }
        .expect("Couldn't bind vertex buffer");

    (vertex_buffer, vertex_buffer_memory)
}

// taken from vulkan-tutorial-rust
fn find_memory_type(
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    for (i, memory_type) in memory_properties.memory_types.iter().enumerate() {
        // this is magic, accept it
        if (type_filter & (1 << i)) > 0 && memory_type.property_flags.contains(required_properties)
        {
            return i as u32;
        }
    }

    panic!("Failed to find suitable memory type!")
}

impl Vertex {
    fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, position) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}

fn create_swapchain(
    physical_device: vk::PhysicalDevice,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    swapchain_creator: &Swapchain,
) -> (vk::SwapchainKHR, Vec<vk::Image>, vk::Extent2D) {
    // check device swapchain capabilties (not just that it has the extension,
    // also formats and stuff like that)
    // also returns what dimensions the swapchain should initially be created at
    let dimensions = check_device_swapchain_caps(surface_loader, physical_device, surface);

    // for now, the format is fixed - might be good to change later.

    // create swapchain
    let create_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: vk::SwapchainCreateFlagsKHR::empty(),
        surface: surface,
        min_image_count: 2,
        image_format: SWAPCHAIN_FORMAT,
        image_color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        image_extent: dimensions,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode: vk::PresentModeKHR::IMMEDIATE,
        clipped: vk::TRUE,
        old_swapchain: vk::SwapchainKHR::null(),
    };

    let swapchain = unsafe { swapchain_creator.create_swapchain(&create_info, None) }
        .expect("Couldn't create swapchain");

    let images = unsafe { swapchain_creator.get_swapchain_images(swapchain) }
        .expect("Couldn't get swapchain images");

    (swapchain, images, dimensions)
}

fn create_swapchain_image_views<D: DeviceV1_0>(
    device: &D,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    images
        .iter()
        .map(|image| {
            let iv_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: SWAPCHAIN_FORMAT,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            unsafe { device.create_image_view(&iv_info, None) }
                .expect("Couldn't create image view info")
        })
        .collect()
}

fn create_pipeline<D: DeviceV1_0>(
    device: &D,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    swapchain_dims: vk::Extent2D,
    shader_stages: [vk::PipelineShaderStageCreateInfo; 2],
    binding_descriptions: &[vk::VertexInputBindingDescription],
    attribute_descriptions: &[vk::VertexInputAttributeDescription],
) -> vk::Pipeline {
    // vertex format
    let pipeline_vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        vertex_binding_description_count: binding_descriptions.len() as u32,
        p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
        vertex_attribute_description_count: attribute_descriptions.len() as u32,
        p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
    };

    let pipeline_input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_dims.width as f32,
        height: swapchain_dims.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];

    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_dims,
    }];

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
    };

    let pipeline_rasterization_info = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::CLOCKWISE,
        depth_bias_enable: vk::FALSE,
        depth_bias_constant_factor: 0.0,
        depth_bias_clamp: 0.0,
        depth_bias_slope_factor: 0.0,
        line_width: 1.0,
    };

    let pipeline_multisample_info = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 1.0,
        p_sample_mask: ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
    };

    // color blending info per framebuffer
    let pipeline_color_blend_attachment_infos = [vk::PipelineColorBlendAttachmentState {
        blend_enable: vk::FALSE,
        // not used because we disabled blending
        src_color_blend_factor: vk::BlendFactor::ONE,
        dst_color_blend_factor: vk::BlendFactor::ZERO,
        color_blend_op: vk::BlendOp::ADD,
        src_alpha_blend_factor: vk::BlendFactor::ONE,
        dst_alpha_blend_factor: vk::BlendFactor::ZERO,
        alpha_blend_op: vk::BlendOp::ADD,

        // is used
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
    }];

    // color blending settings for the whole pipleine
    let pipeline_color_blend_info = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY, // optional
        attachment_count: pipeline_color_blend_attachment_infos.len() as u32,
        p_attachments: pipeline_color_blend_attachment_infos.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0], // optional
    };

    let pipeline_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &pipeline_vertex_input_info,
        p_input_assembly_state: &pipeline_input_assembly_info,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &pipeline_rasterization_info,
        p_multisample_state: &pipeline_multisample_info,
        p_depth_stencil_state: ptr::null(),
        p_color_blend_state: &pipeline_color_blend_info,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: 0,
    }];

    unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None) }
        .expect("Couldn't create graphics pipeline")[0]
}

fn create_pipeline_layout<D: DeviceV1_0>(device: &D) -> vk::PipelineLayout {
    // fixed-function pipeline settings

    // we don't use any shader uniforms so we leave it empty
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: 0,
        p_set_layouts: ptr::null(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
    };

    unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .expect("Couldn't create pipeline layout!")
    }
}

fn cleanup_swapchain<D: DeviceV1_0>(
    device: &D,
    swapchain_creator: &Swapchain,
    framebuffers: Vec<vk::Framebuffer>,
    pipeline: vk::Pipeline,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain: vk::SwapchainKHR,
) {
    unsafe {
        swapchain_image_views
            .iter()
            .for_each(|iv| device.destroy_image_view(*iv, None));
        framebuffers
            .iter()
            .for_each(|fb| device.destroy_framebuffer(*fb, None));
        device.destroy_pipeline(pipeline, None);
        swapchain_creator.destroy_swapchain(swapchain, None);
    }
}

fn setup_flying_frames<D: DeviceV1_0>(
    device: &D,
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    vertex_buffer_capacity: DeviceSize,
    index_buffer_capacity: DeviceSize,
) -> Vec<FlyingFrame> {
    // create info for semaphores and fences
    let semaphore_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
    };

    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::FenceCreateFlags::SIGNALED,
    };

    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            let (vertex_staging_buffer, vertex_staging_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                vertex_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let (index_staging_buffer, index_staging_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                index_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );

            let (vertex_buffer, vertex_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                vertex_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let (index_buffer, index_buffer_memory) = create_buffer(
                device,
                device_memory_properties,
                index_buffer_capacity,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let image_available_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .expect("Couldn't create semaphore");

            let render_finished_semaphore =
                unsafe { device.create_semaphore(&semaphore_info, None) }
                    .expect("Couldn't create semaphore");

            let render_finished_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            let vbuf_copied_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            let ibuf_copied_fence =
                unsafe { device.create_fence(&fence_info, None) }.expect("Couldn't create fence");

            FlyingFrame {
                image_available_semaphore,
                render_finished_semaphore,
                render_finished_fence,
                command_buffer: None,

                vertex_staging_buffer,
                vertex_staging_buffer_memory,
                vertex_buffer,
                vertex_buffer_memory,
                index_staging_buffer,
                index_staging_buffer_memory,
                index_buffer,
                index_buffer_memory,

                vbuf_copied_fence,
                ibuf_copied_fence,
            }
        })
        .collect()
}

fn create_command_pool<D: DeviceV1_0>(device: &D, queue_family_index: u32) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::CommandPoolCreateFlags::empty(),
        queue_family_index,
    };

    unsafe { device.create_command_pool(&command_pool_info, None) }
        .expect("Couldn't create command pool")
}

fn create_command_buffer<D: DeviceV1_0>(
    device: &D,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    framebuffer: vk::Framebuffer,
    dimensions: vk::Extent2D,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    index_count: u32,
) -> vk::CommandBuffer {
    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_pool,
        // Primary: can be submitted to a queue, but not called from other command buffers
        // Secondary: can't be directly submitted, but can be called from other command buffers
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
    };

    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_alloc_info) }
        .expect("Couldn't allocate command buffers")[0];

    // begin command buffer
    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        flags: vk::CommandBufferUsageFlags::empty(),
        p_inheritance_info: ptr::null(),
    };

    unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
        .expect("Couldn't begin command buffer");

    // Start render pass
    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.5, 0.5, 0.5, 1.0],
        },
    }];

    let render_pass_begin_info = vk::RenderPassBeginInfo {
        s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
        p_next: ptr::null(),
        render_pass,
        framebuffer,
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: dimensions,
        },
        clear_value_count: 1,
        p_clear_values: clear_values.as_ptr(),
    };

    unsafe {
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        // bind vertex buffer
        let vertex_buffers = [vertex_buffer];
        let offsets: [vk::DeviceSize; 1] = [0];
        device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

        // bind index buffer
        device.cmd_bind_index_buffer(command_buffer, index_buffer, 0, vk::IndexType::UINT32);

        // <index_count> indices, 1 instance, first index 0, vertex offset 0,
        // first instance 0
        device.cmd_draw_indexed(command_buffer, index_count, 1, 0, 0, 0);

        device.cmd_end_render_pass(command_buffer);
    }

    unsafe { device.end_command_buffer(command_buffer) }.expect("Couldn't record command buffer!");

    command_buffer
}

fn create_framebuffers<D: DeviceV1_0>(
    device: &D,
    render_pass: vk::RenderPass,
    dimensions: vk::Extent2D,
    image_views: &[vk::ImageView],
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|iv| {
            let image_views = [*iv];
            let framebuffer_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                render_pass,
                attachment_count: 1,
                p_attachments: image_views.as_ptr(),
                width: dimensions.width,
                height: dimensions.height,
                layers: 1,
            };

            unsafe { device.create_framebuffer(&framebuffer_info, None) }
                .expect("Couldn't create framebuffer")
        })
        .collect()
}

fn create_shader_module<D: DeviceV1_0>(device: &D, code: Vec<u8>) -> vk::ShaderModule {
    use ash::util::read_spv;
    use std::io::Cursor;

    let readable_code = read_spv(&mut Cursor::new(&code)).expect("Couldn't read SPV");
    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&readable_code);

    unsafe {
        device
            .create_shader_module(&shader_module_create_info, None)
            .expect("Couldn't create shader module")
    }
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

#[cfg(target_os = "macos")]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        MacOSSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}

#[cfg(all(windows))]
fn extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
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
    eprintln!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

fn is_phys_dev_suitable(instance: &ash::Instance, phys_dev: &vk::PhysicalDevice) -> bool {
    // gets a list of extensions supported by this device as vulkan strings,
    // which don't implement PartialEq
    let extension_properties = unsafe { instance.enumerate_device_extension_properties(*phys_dev) }
        .expect("Couldn't enumerate device extension properties!");

    // Now convert them into rust strings
    let available_extension_names: Vec<String> = extension_properties
        .iter()
        .map(|ext| vk_to_string(&ext.extension_name))
        .collect();

    // make sure all required device extensions are supported by this device
    get_device_extensions().iter().for_each(|name| {
        available_extension_names
            .iter()
            .find(|ext| ext == name)
            .expect(&format!("Couldn't find extension {}", name));
    });

    true
}

fn check_device_swapchain_caps(
    surface_loader: &Surface,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::Extent2D {
    // returns the current dimensions of the swapchain

    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .expect("Couldn't get physical device surface capabilities");

    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
            .expect("Couldn't get physical device surface formats");

    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }
    .expect("Couldn't get physical device surface present modes");

    // we will request 3 swapchain images to avoid having to wait while one is
    // being cleared or something, idk exactly
    assert!(capabilities.min_image_count <= 3 && capabilities.max_image_count >= 3);

    formats
        .iter()
        .find(|fmt| fmt.format == vk::Format::B8G8R8A8_UNORM)
        .expect("Swapchain doesn't support B8G8R8A8_UNORM!");

    assert!(present_modes.contains(&vk::PresentModeKHR::IMMEDIATE));

    capabilities.current_extent
}

// many of these functions are ripped from https://github.com/bwasty/vulkan-tutorial-rs

// this is ripped from the ash examples
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::os::unix::WindowExt;
    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winit::os::macos::WindowExt;

    let wnd: cocoa_id = mem::transmute(window.get_nswindow());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.get_nsview() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}

#[cfg(target_os = "windows")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::ptr;
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::os::windows::WindowExt;

    let hwnd = window.get_hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}

fn create_render_pass(device: &ash::Device) -> vk::RenderPass {
    // our render pass has a single image, so only one attachment description is
    // necessary
    let attachment_descs = [vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: SWAPCHAIN_FORMAT,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    }];

    let color_attachments = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpass_descs = [vk::SubpassDescription {
        flags: vk::SubpassDescriptionFlags::empty(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: 1,
        p_color_attachments: color_attachments.as_ptr(),
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];

    // apparently needed to ensure we don't start drawing before we acquire a
    // swapchain image, but I don't know why it's necessary because we already
    // use a semaphore to synchronize between acquiring the image and executing
    // the command buffer.
    let subpass_dependencies = [vk::SubpassDependency {
        // "The special value VK_SUBPASS_EXTERNAL refers to the implicit subpass
        // before or after the render pass depending on whether it is specified
        // in srcSubpass or dstSubpass"
        src_subpass: vk::SUBPASS_EXTERNAL,

        // refers to the subpass we draw in, which is at index 0
        dst_subpass: 0,

        // i don't understand this
        src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        src_access_mask: vk::AccessFlags::empty(),
        dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dependency_flags: vk::DependencyFlags::empty(),
    }];

    let render_pass_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::RenderPassCreateFlags::empty(),
        attachment_count: 1,
        p_attachments: attachment_descs.as_ptr(),
        subpass_count: 1,
        p_subpasses: subpass_descs.as_ptr(),
        dependency_count: 1,
        p_dependencies: subpass_dependencies.as_ptr(),
    };

    unsafe {
        device
            .create_render_pass(&render_pass_info, None)
            .expect("Failed to create render pass!")
    }
}

// Helper function to convert [c_char; SIZE] to string
fn vk_to_string(raw_string_array: &[c_char]) -> String {
    // Implementation 2
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

fn get_device_extensions<'a>() -> [&'a str; 1] {
    ["VK_KHR_swapchain"]
}

fn get_device_extensions_raw() -> [*const c_char; 1] {
    [ash::extensions::khr::Swapchain::name().as_ptr()]
}

fn read_shader_code(shader_path: &Path) -> Vec<u8> {
    use std::fs::File;
    use std::io::Read;

    let spv_file =
        File::open(shader_path).expect(&format!("Failed to find spv file at {:?}", shader_path));
    let bytes_code: Vec<u8> = spv_file.bytes().filter_map(|byte| byte.ok()).collect();

    bytes_code
}

fn pixel_to_vk(screen_dims: vk::Extent2D, pixel_pos: &PixelPos) -> VkPos {
    let screen_width = screen_dims.width as f64;
    let screen_height = screen_dims.height as f64;

    [
        (pixel_pos[0] - screen_width * 0.5) / screen_width * 2.0,
        (pixel_pos[1] - screen_height * 0.5) / screen_height * 2.0,
    ]
}

fn vk_to_point(vk_pos: &VkPos) -> lyon::math::Point {
    lyon::math::point(vk_pos[0] as f32, vk_pos[1] as f32)
}

fn pixel_dist(a: &PixelPos, b: &PixelPos) -> f64 {
    ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])).sqrt()
}
