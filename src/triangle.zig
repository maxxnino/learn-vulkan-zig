const std = @import("std");
const vk = @import("vulkan");
const c = @import("c.zig");
const resources = @import("resources");
const GraphicsContext = @import("graphics_context.zig").GraphicsContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;
const za = @import("zalgebra");
const Mat4 = za.Mat4;
const Vec3 = za.Vec3;
const assert = std.debug.assert;

const app_name = "vulkan-zig triangle example";

const Vertex = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };

    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "color"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_sfloat,
            .offset = @offsetOf(Vertex, "tex_coord"),
        },
    };

    pos: [3]f32,
    color: [3]f32,
    tex_coord: [2]f32,
};

const UniformBufferObject = struct {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
};

fn setViewDirection(position: Vec3, direction: Vec3, up: Vec3) Mat4 {
    const w = direction.norm();
    const u = w.cross(up).norm();
    const v = w.cross(u);

    var view = Mat4.identity();
    view.data[0][0] = u.x;
    view.data[1][0] = u.y;
    view.data[2][0] = u.z;
    view.data[0][1] = v.x;
    view.data[1][1] = v.y;
    view.data[2][1] = v.z;
    view.data[0][2] = w.x;
    view.data[1][2] = w.y;
    view.data[2][2] = w.z;
    view.data[3][0] = u.dot(position);
    view.data[3][1] = v.dot(position);
    view.data[3][2] = w.dot(position);
    return view;
}

fn setViewTarget(position: Vec3, target: Vec3, up: Vec3) Mat4 {
    return setViewDirection(position, target.sub(position), up);
}

const CameraPos = struct {
    rotation: Vec3,
    translation: Vec3,
    xpos: f64,
    ypos: f64,
    const look_speed: f32 = 1.5;
    const move_speed: f32 = 8;

    fn moveInPlaneXZ(self: *CameraPos, window: *c.GLFWwindow, dt: f32) void {
        const prev_xpos = self.xpos;
        const prev_ypos = self.ypos;
        c.glfwGetCursorPos(window, &self.xpos, &self.ypos);
        const ydelta = @floatCast(f32, self.ypos - prev_ypos);
        const xdelta = @floatCast(f32, self.xpos - prev_xpos);
        var rotate = Vec3.new(ydelta, -xdelta, 0);
        // if (ydelta > std.math.epsilon(f64) and xdelta > std.math.epsilon(f64))
        // else
        //     Vec3.zero();

        if (rotate.dot(rotate) > std.math.epsilon(f32)) {
            self.rotation = self.rotation.add(rotate.norm().scale(look_speed * dt));
        }

        // limit pitch values between about +/- 85ish degrees
        self.rotation.x = std.math.clamp(self.rotation.x, -1.5, 1.5);
        self.rotation.y = std.math.mod(f32, self.rotation.y, 2 * std.math.pi) catch unreachable;

        const yaw = self.rotation.y;
        const forward_dir = Vec3.new(std.math.sin(yaw), 0, std.math.cos(yaw));
        const right_dir = Vec3.new(forward_dir.z, 0, -forward_dir.x);
        const up_dir = Vec3.up();

        var move_dir = Vec3.zero();
        if (c.glfwGetKey(window, c.GLFW_KEY_W) == c.GLFW_PRESS) move_dir = move_dir.add(forward_dir);
        if (c.glfwGetKey(window, c.GLFW_KEY_S) == c.GLFW_PRESS) move_dir = move_dir.sub(forward_dir);
        if (c.glfwGetKey(window, c.GLFW_KEY_A) == c.GLFW_PRESS) move_dir = move_dir.add(right_dir);
        if (c.glfwGetKey(window, c.GLFW_KEY_D) == c.GLFW_PRESS) move_dir = move_dir.sub(right_dir);
        if (c.glfwGetKey(window, c.GLFW_KEY_Z) == c.GLFW_PRESS) move_dir = move_dir.add(up_dir);
        if (c.glfwGetKey(window, c.GLFW_KEY_C) == c.GLFW_PRESS) move_dir = move_dir.sub(up_dir);

        if (move_dir.dot(move_dir) > std.math.epsilon(f32)) {
            self.translation = self.translation.add(move_dir.norm().scale(move_speed * dt));
        }
    }
    fn getViewYXZ(position: Vec3, rotation: Vec3) Mat4 {
        const c3 = std.math.cos(rotation.z);
        const s3 = std.math.sin(rotation.z);
        const c2 = std.math.cos(rotation.x);
        const s2 = std.math.sin(rotation.x);
        const c1 = std.math.cos(rotation.y);
        const s1 = std.math.sin(rotation.y);
        const u = Vec3.new((c1 * c3 + s1 * s2 * s3), (c2 * s3), (c1 * s2 * s3 - c3 * s1));
        const v = Vec3.new((c3 * s1 * s2 - c1 * s3), (c2 * c3), (c1 * c3 * s2 + s1 * s3));
        const w = Vec3.new((c2 * s1), (-s2), (c1 * c2));
        var view_matrix = Mat4.identity();
        view_matrix.data[0][0] = u.x;
        view_matrix.data[1][0] = u.y;
        view_matrix.data[2][0] = u.z;
        view_matrix.data[0][1] = v.x;
        view_matrix.data[1][1] = v.y;
        view_matrix.data[2][1] = v.z;
        view_matrix.data[0][2] = w.x;
        view_matrix.data[1][2] = w.y;
        view_matrix.data[2][2] = w.z;
        view_matrix.data[3][0] = u.dot(position);
        view_matrix.data[3][1] = v.dot(position);
        view_matrix.data[3][2] = w.dot(position);
        return view_matrix;
    }
};

const vertices = [_]Vertex{
    .{ .pos = .{ -0.5, -0.5, 0 }, .color = .{ 1, 0, 0 }, .tex_coord = .{ 1, 0 } },
    .{ .pos = .{ 0.5, -0.5, 0 }, .color = .{ 0, 1, 0 }, .tex_coord = .{ 0, 0 } },
    .{ .pos = .{ 0.5, 0.5, 0 }, .color = .{ 0, 0, 1 }, .tex_coord = .{ 0, 1 } },
    .{ .pos = .{ -0.5, 0.5, 0 }, .color = .{ 1, 1, 1 }, .tex_coord = .{ 1, 1 } },
    .{ .pos = .{ -0.5, -0.5, -0.5 }, .color = .{ 1, 0, 0 }, .tex_coord = .{ 1, 0 } },
    .{ .pos = .{ 0.5, -0.5, -0.5 }, .color = .{ 0, 1, 0 }, .tex_coord = .{ 0, 0 } },
    .{ .pos = .{ 0.5, 0.5, -0.5 }, .color = .{ 0, 0, 1 }, .tex_coord = .{ 0, 1 } },
    .{ .pos = .{ -0.5, 0.5, -0.5 }, .color = .{ 1, 1, 1 }, .tex_coord = .{ 1, 1 } },
};
const indices = [_]u16{ 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 };

const BufferMemory = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,

    pub fn init(
        gc: GraphicsContext,
        size: vk.DeviceSize,
        usage: vk.BufferUsageFlags,
        properties: vk.MemoryPropertyFlags,
    ) !BufferMemory {
        var bm: BufferMemory = undefined;
        bm.buffer = try gc.vkd.createBuffer(gc.dev, .{
            .flags = .{},
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        }, null);
        const mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, bm.buffer);
        bm.memory = try gc.allocate(mem_reqs, properties);
        try gc.vkd.bindBufferMemory(gc.dev, bm.buffer, bm.memory, 0);
        return bm;
    }

    pub fn deinit(self: BufferMemory, gc: GraphicsContext) void {
        gc.vkd.freeMemory(gc.dev, self.memory, null);
        gc.vkd.destroyBuffer(gc.dev, self.buffer, null);
    }
};
const TextureImage = struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,
    sampler: vk.Sampler,

    pub fn init(
        gc: GraphicsContext,
        width: u32,
        height: u32,
        format: vk.Format,
        tiling: vk.ImageTiling,
        usage: vk.ImageUsageFlags,
        properties: vk.MemoryPropertyFlags,
    ) !TextureImage {
        var bm: TextureImage = undefined;
        const ici = vk.ImageCreateInfo{
            .flags = .{},
            .image_type = .@"2d",
            .format = format,
            .extent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = tiling,
            .usage = usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .initial_layout = .@"undefined",
        };
        bm.image = try gc.vkd.createImage(gc.dev, ici, null);
        errdefer gc.vkd.destroyImage(gc.dev, bm.image, null);

        const mem_reqs = gc.vkd.getImageMemoryRequirements(gc.dev, bm.image);
        bm.memory = try gc.allocate(mem_reqs, properties);
        errdefer gc.vkd.freeMemory(gc.dev, bm.memory, null);

        try gc.vkd.bindImageMemory(gc.dev, bm.image, bm.memory, 0);

        bm.view = try gc.vkd.createImageView(gc.dev, .{
            .flags = .{},
            .image = bm.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer gc.vkd.destroyImageView(gc.dev, bm.view, null);

        const sci = vk.SamplerCreateInfo{
            .flags = .{},
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_mode = .linear,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = vk.TRUE,
            .max_anisotropy = gc.props.limits.max_sampler_anisotropy,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = 0,
            .border_color = .int_opaque_black,
            .unnormalized_coordinates = vk.FALSE,
        };
        bm.sampler = try gc.vkd.createSampler(gc.dev, sci, null);
        return bm;
    }

    pub fn deinit(self: TextureImage, gc: GraphicsContext) void {
        gc.vkd.freeMemory(gc.dev, self.memory, null);
        gc.vkd.destroyImage(gc.dev, self.image, null);
        gc.vkd.destroyImageView(gc.dev, self.view, null);
        gc.vkd.destroySampler(gc.dev, self.sampler, null);
    }
};
const DepthImage = struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,

    pub fn init(
        gc: GraphicsContext,
        width: u32,
        height: u32,
        format: vk.Format,
        tiling: vk.ImageTiling,
        usage: vk.ImageUsageFlags,
        properties: vk.MemoryPropertyFlags,
    ) !DepthImage {
        var bm: DepthImage = undefined;
        const ici = vk.ImageCreateInfo{
            .flags = .{},
            .image_type = .@"2d",
            .format = format,
            .extent = .{
                .width = width,
                .height = height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = tiling,
            .usage = usage,
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .initial_layout = .@"undefined",
        };
        bm.image = try gc.vkd.createImage(gc.dev, ici, null);
        errdefer gc.vkd.destroyImage(gc.dev, bm.image, null);

        const mem_reqs = gc.vkd.getImageMemoryRequirements(gc.dev, bm.image);
        bm.memory = try gc.allocate(mem_reqs, properties);
        errdefer gc.vkd.freeMemory(gc.dev, bm.memory, null);

        try gc.vkd.bindImageMemory(gc.dev, bm.image, bm.memory, 0);

        bm.view = try gc.vkd.createImageView(gc.dev, .{
            .flags = .{},
            .image = bm.image,
            .view_type = .@"2d",
            .format = format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                // NOTE: different compare to Swapchain image and TextureImage
                .aspect_mask = .{ .depth_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        return bm;
    }

    pub fn deinit(self: DepthImage, gc: GraphicsContext) void {
        gc.vkd.freeMemory(gc.dev, self.memory, null);
        gc.vkd.destroyImage(gc.dev, self.image, null);
        gc.vkd.destroyImageView(gc.dev, self.view, null);
    }

    pub fn findDepthFormat(gc: GraphicsContext) ?vk.Format {
        return gc.findSupportedFormat(
            &.{ .d32_sfloat, .d32_sfloat_s8_uint, .d24_unorm_s8_uint },
            .optimal,
            .{ .depth_stencil_attachment_bit = true },
        );
    }
    pub fn hasStencilComponent(format: vk.Format) bool {
        return format == .d32_sfloat_s8_uint or format == .d24_unorm_s8_uint;
    }
};
pub fn main() !void {
    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window = c.glfwCreateWindow(
        @intCast(c_int, extent.width),
        @intCast(c_int, extent.height),
        app_name,
        null,
        null,
    ) orelse return error.WindowInitFailed;
    defer c.glfwDestroyWindow(window);

    const allocator = std.heap.page_allocator;

    const gc = try GraphicsContext.init(allocator, app_name, window);
    defer gc.deinit();

    std.debug.print("Using device: {s}\n", .{gc.deviceName()});

    var swapchain = try Swapchain.init(&gc, allocator, extent);
    defer swapchain.deinit();

    const render_pass = try createRenderPass(gc, swapchain);
    defer gc.vkd.destroyRenderPass(gc.dev, render_pass, null);

    const descriptor_layout = try createDescriptorSetLayout(gc);
    defer gc.vkd.destroyDescriptorSetLayout(gc.dev, descriptor_layout, null);

    const pipeline_layout = try gc.vkd.createPipelineLayout(gc.dev, .{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &descriptor_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer gc.vkd.destroyPipelineLayout(gc.dev, pipeline_layout, null);

    var pipeline = try createPipeline(&gc, pipeline_layout, render_pass);
    defer gc.vkd.destroyPipeline(gc.dev, pipeline, null);

    var depth_image = try createDepthResources(gc, swapchain.extent);
    defer depth_image.deinit(gc);

    var framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain, depth_image);
    defer destroyFramebuffers(&gc, allocator, framebuffers);

    const pool = try gc.vkd.createCommandPool(gc.dev, .{
        .flags = .{},
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.vkd.destroyCommandPool(gc.dev, pool, null);

    const vertex_buffer = try createVertexBuffer(gc, pool);
    defer vertex_buffer.deinit(gc);

    const index_buffer = try createIndexBuffer(gc, pool);
    defer index_buffer.deinit(gc);

    var unibufs = try createUniformBuffer(gc, allocator, framebuffers);
    defer destroyUniformBuffers(gc, allocator, unibufs);

    var descriptor_pool = try createDescriptorPool(gc, framebuffers);
    defer gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);

    const texture = try createTextureImage(gc, pool);
    defer texture.deinit(gc);
    var descriptor_sets = try createDescriptorSets(
        gc,
        allocator,
        descriptor_pool,
        descriptor_layout,
        unibufs,
        texture,
    );
    defer allocator.free(descriptor_sets);

    var cmdbufs = try createCommandBuffers(
        &gc,
        pool,
        allocator,
        vertex_buffer.buffer,
        index_buffer.buffer,
        swapchain.extent,
        render_pass,
        pipeline,
        framebuffers,
        pipeline_layout,
        descriptor_sets,
    );
    defer destroyCommandBuffers(&gc, pool, allocator, cmdbufs);
    var update_timer = try std.time.Timer.start();
    var camera = CameraPos{
        .rotation = Vec3.zero(),
        .translation = Vec3.new(0, 0, -2),
        .xpos = 0,
        .ypos = 0,
    };
    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        const dt = @intToFloat(f32, update_timer.lap()) / @intToFloat(f32, std.time.ns_per_s);
        const cmdbuf = cmdbufs[swapchain.image_index];
        const unibuf = unibufs[swapchain.image_index];
        camera.moveInPlaneXZ(window, dt);
        try updateUniformBuffer(
            gc,
            unibuf,
            swapchain.extent,
            CameraPos.getViewYXZ(camera.translation, camera.rotation),
        );
        //TODO: chapter 2 descriptor set

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        var w: c_int = undefined;
        var h: c_int = undefined;
        c.glfwGetWindowSize(window, &w, &h);

        if (state == .suboptimal or extent.width != @intCast(u32, w) or extent.height != @intCast(u32, h)) {
            extent.width = @intCast(u32, w);
            extent.height = @intCast(u32, h);
            try swapchain.recreate(extent);

            depth_image.deinit(gc);
            depth_image = try createDepthResources(gc, swapchain.extent);

            destroyFramebuffers(&gc, allocator, framebuffers);
            framebuffers = try createFramebuffers(&gc, allocator, render_pass, swapchain, depth_image);

            destroyUniformBuffers(gc, allocator, unibufs);
            unibufs = try createUniformBuffer(gc, allocator, framebuffers);

            gc.vkd.destroyDescriptorPool(gc.dev, descriptor_pool, null);
            descriptor_pool = try createDescriptorPool(gc, framebuffers);

            allocator.free(descriptor_sets);
            descriptor_sets = try createDescriptorSets(
                gc,
                allocator,
                descriptor_pool,
                descriptor_layout,
                unibufs,
                texture,
            );

            destroyCommandBuffers(&gc, pool, allocator, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &gc,
                pool,
                allocator,
                vertex_buffer.buffer,
                index_buffer.buffer,
                swapchain.extent,
                render_pass,
                pipeline,
                framebuffers,
                pipeline_layout,
                descriptor_sets,
            );
        }

        c.glfwPollEvents();
    }

    try swapchain.waitForAllFences();
}
fn updateUniformBuffer(gc: GraphicsContext, buffer: BufferMemory, extent: vk.Extent2D, view: Mat4) !void {
    var proj = za.perspective(
        45.0,
        @intToFloat(f32, extent.width) / @intToFloat(f32, extent.height),
        0.1,
        100.0,
    );
    const ubo = UniformBufferObject{
        .proj = proj,
        .view = view,
        // .model = Mat4.identity().rotate(90, Vec3.new(0, 0, 1)),
        .model = Mat4.identity(),
    };

    {
        const data = try gc.vkd.mapMemory(gc.dev, buffer.memory, 0, @sizeOf(UniformBufferObject), .{});
        defer gc.vkd.unmapMemory(gc.dev, buffer.memory);

        const gpu_memory = @ptrCast(*UniformBufferObject, @alignCast(@alignOf(UniformBufferObject), data));
        gpu_memory.* = ubo;
    }
}
fn createIndexBuffer(gc: GraphicsContext, pool: vk.CommandPool) !BufferMemory {
    const size = @sizeOf(@TypeOf(indices));
    var buffer = try BufferMemory.init(
        gc,
        size,
        .{ .transfer_dst_bit = true, .index_buffer_bit = true },
        .{ .device_local_bit = true },
    );
    var stage_buffer = try BufferMemory.init(
        gc,
        size,
        .{ .transfer_src_bit = true },
        .{ .host_coherent_bit = true, .host_visible_bit = true },
    );
    defer stage_buffer.deinit(gc);

    // Copy vertices to stage buffer
    {
        const data = try gc.vkd.mapMemory(gc.dev, stage_buffer.memory, 0, vk.WHOLE_SIZE, .{});
        defer gc.vkd.unmapMemory(gc.dev, stage_buffer.memory);

        const gpu_memory = @ptrCast([*]u16, @alignCast(@alignOf(u16), data));
        for (indices) |indice, i| {
            gpu_memory[i] = indice;
        }
    }

    // Copy containt form stage buffer to vertex buffer
    try copyBuffer(gc, pool, buffer.buffer, stage_buffer.buffer, size);
    return buffer;
}

fn createVertexBuffer(gc: GraphicsContext, pool: vk.CommandPool) !BufferMemory {
    const size = @sizeOf(@TypeOf(vertices));
    var vertex_buffer = try BufferMemory.init(
        gc,
        size,
        .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .{ .device_local_bit = true },
    );
    var stage_buffer = try BufferMemory.init(
        gc,
        size,
        .{ .transfer_src_bit = true },
        .{ .host_coherent_bit = true, .host_visible_bit = true },
    );
    defer stage_buffer.deinit(gc);

    // Copy vertices to stage buffer
    {
        const data = try gc.vkd.mapMemory(gc.dev, stage_buffer.memory, 0, vk.WHOLE_SIZE, .{});
        defer gc.vkd.unmapMemory(gc.dev, stage_buffer.memory);

        const gpu_vertices = @ptrCast([*]Vertex, @alignCast(@alignOf(Vertex), data));
        for (vertices) |vertex, i| {
            gpu_vertices[i] = vertex;
        }
    }

    // Copy containt form stage buffer to vertex buffer
    try copyBuffer(gc, pool, vertex_buffer.buffer, stage_buffer.buffer, size);
    return vertex_buffer;
}

fn copyBuffer(gc: GraphicsContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    const cmdbuf = try beginSingleTimeCommand(gc, pool);
    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    gc.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast([*]const vk.BufferCopy, &region));
    try endSingleTimeCommands(gc, pool, cmdbuf);
}

fn createCommandBuffers(
    gc: *const GraphicsContext,
    pool: vk.CommandPool,
    allocator: *Allocator,
    buffer: vk.Buffer,
    index_buffer: vk.Buffer,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    framebuffers: []vk.Framebuffer,
    pipeline_layout: vk.PipelineLayout,
    descriptor_sets: []vk.DescriptorSet,
) ![]vk.CommandBuffer {
    const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(cmdbufs);

    try gc.vkd.allocateCommandBuffers(gc.dev, .{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = @truncate(u32, cmdbufs.len),
    }, cmdbufs.ptr);
    errdefer gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(u32, cmdbufs.len), cmdbufs.ptr);

    const clear = [_]vk.ClearValue{
        .{
            .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
        },
        .{
            .depth_stencil = .{ .depth = 1, .stencil = 0 },
        },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @intToFloat(f32, extent.width),
        .height = @intToFloat(f32, extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    for (cmdbufs) |cmdbuf, i| {
        try gc.vkd.beginCommandBuffer(cmdbuf, .{
            .flags = .{},
            .p_inheritance_info = null,
        });

        gc.vkd.cmdSetViewport(cmdbuf, 0, 1, @ptrCast([*]const vk.Viewport, &viewport));
        gc.vkd.cmdSetScissor(cmdbuf, 0, 1, @ptrCast([*]const vk.Rect2D, &scissor));

        gc.vkd.cmdBeginRenderPass(cmdbuf, .{
            .render_pass = render_pass,
            .framebuffer = framebuffers[i],
            .render_area = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = extent,
            },
            .clear_value_count = @truncate(u32, clear.len),
            .p_clear_values = @ptrCast([*]const vk.ClearValue, &clear),
        }, .@"inline");

        gc.vkd.cmdBindPipeline(cmdbuf, .graphics, pipeline);
        const offset = [_]vk.DeviceSize{0};
        gc.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast([*]const vk.Buffer, &buffer), &offset);
        gc.vkd.cmdBindIndexBuffer(cmdbuf, index_buffer, 0, .uint16);
        gc.vkd.cmdBindDescriptorSets(
            cmdbuf,
            .graphics,
            pipeline_layout,
            0,
            1,
            @ptrCast([*]const vk.DescriptorSet, &descriptor_sets[i]),
            0,
            undefined,
        );
        gc.vkd.cmdDrawIndexed(cmdbuf, @truncate(u32, indices.len), 1, 0, 0, 0);
        gc.vkd.cmdEndRenderPass(cmdbuf);
        try gc.vkd.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(
    gc: *const GraphicsContext,
    pool: vk.CommandPool,
    allocator: *Allocator,
    cmdbufs: []vk.CommandBuffer,
) void {
    gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(u32, cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

fn createFramebuffers(
    gc: *const GraphicsContext,
    allocator: *Allocator,
    render_pass: vk.RenderPass,
    swapchain: Swapchain,
    depth_image: DepthImage,
) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb| {
        const attachments = [_]vk.ImageView{
            swapchain.swap_images[i].view,
            depth_image.view,
        };
        fb.* = try gc.vkd.createFramebuffer(gc.dev, .{
            .flags = .{},
            .render_pass = render_pass,
            .attachment_count = @truncate(u32, attachments.len),
            .p_attachments = @ptrCast([*]const vk.ImageView, &attachments),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, allocator: *Allocator, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);
    allocator.free(framebuffers);
}

fn createRenderPass(gc: GraphicsContext, swapchain: Swapchain) !vk.RenderPass {
    const attachments = [_]vk.AttachmentDescription{
        // color attachment
        .{
            .flags = .{},
            .format = swapchain.surface_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .present_src_khr,
        },
        //depth_attachment
        .{
            .flags = .{},
            .format = DepthImage.findDepthFormat(gc).?,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .@"undefined",
            .final_layout = .depth_stencil_attachment_optimal,
        },
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const depth_attachment_ref = vk.AttachmentReference{
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast([*]const vk.AttachmentReference, &color_attachment_ref),
        .p_resolve_attachments = null,
        .p_depth_stencil_attachment = &depth_attachment_ref,
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };
    const sd = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true, .early_fragment_tests_bit = true },
        .src_access_mask = .{},
        .dst_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
        .dependency_flags = .{},
    };
    return try gc.vkd.createRenderPass(gc.dev, .{
        .flags = .{},
        .attachment_count = @truncate(u32, attachments.len),
        .p_attachments = @ptrCast([*]const vk.AttachmentDescription, &attachments),
        .subpass_count = 1,
        .p_subpasses = @ptrCast([*]const vk.SubpassDescription, &subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast([*]const vk.SubpassDependency, &sd),
    }, null);
}

fn createPipeline(
    gc: *const GraphicsContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try gc.vkd.createShaderModule(gc.dev, .{
        .flags = .{},
        .code_size = resources.triangle_vert.len,
        .p_code = @ptrCast([*]const u32, resources.triangle_vert),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, vert, null);

    const frag = try gc.vkd.createShaderModule(gc.dev, .{
        .flags = .{},
        .code_size = resources.triangle_frag.len,
        .p_code = @ptrCast([*]const u32, resources.triangle_frag),
    }, null);
    defer gc.vkd.destroyShaderModule(gc.dev, frag, null);

    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .flags = .{},
            .stage = .{ .vertex_bit = true },
            .module = vert,
            .p_name = "main",
            // There is one more (optional) member, pSpecializationInfo, which we won't be using here,
            // but is worth discussing. It allows you to specify values for shader constants.
            // You can use a single shader module where its behavior can be configured at pipeline
            // creation by specifying different values for the constants used in it. This is more efficient
            // than configuring the shader using variables at render time, because the compiler can
            // do optimizations like eliminating if statements that depend on these values
            .p_specialization_info = null,
        },
        .{
            .flags = .{},
            .stage = .{ .fragment_bit = true },
            .module = frag,
            .p_name = "main",
            .p_specialization_info = null,
        },
    };

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .flags = .{},
        .vertex_binding_description_count = 1,
        .p_vertex_binding_descriptions = @ptrCast([*]const vk.VertexInputBindingDescription, &Vertex.binding_description),
        .vertex_attribute_description_count = Vertex.attribute_description.len,
        .p_vertex_attribute_descriptions = &Vertex.attribute_description,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .flags = .{},
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .flags = .{},
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    // https://vulkan-tutorial.com/en/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions#page_Rasterizer
    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .flags = .{},
        // If depthClampEnable is set to VK_TRUE, then fragments that are beyond the near and far planes
        // are clamped to them as opposed to discarding them. This is useful in some special cases like shadow maps.
        // Using this requires enabling a GPU feature
        .depth_clamp_enable = vk.FALSE,
        // If rasterizerDiscardEnable is set to VK_TRUE, then geometry never passes through the rasterizer stage.
        // This basically disables any output to the framebuffer.
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .flags = .{},
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .flags = .{},
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @ptrCast([*]const vk.PipelineColorBlendAttachmentState, &pcbas),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const pdssci = vk.PipelineDepthStencilStateCreateInfo{
        .flags = .{},
        .depth_test_enable = vk.TRUE,
        .depth_write_enable = vk.TRUE,
        .depth_compare_op = .less,
        .depth_bounds_test_enable = vk.FALSE,
        .stencil_test_enable = vk.FALSE,
        .front = undefined,
        .back = undefined,
        .min_depth_bounds = 0,
        .max_depth_bounds = 1,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = 2,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = &pdssci,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try gc.vkd.createGraphicsPipelines(
        gc.dev,
        .null_handle,
        1,
        @ptrCast([*]const vk.GraphicsPipelineCreateInfo, &gpci),
        null,
        @ptrCast([*]vk.Pipeline, &pipeline),
    );
    return pipeline;
}
fn createDescriptorSetLayout(gc: GraphicsContext) !vk.DescriptorSetLayout {
    const dslb = [2]vk.DescriptorSetLayoutBinding{
        .{
            .binding = 0,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true },
            .p_immutable_samplers = null,
        },
        .{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
            .descriptor_count = 1,
            .stage_flags = .{ .fragment_bit = true },
            .p_immutable_samplers = null,
        },
    };
    return try gc.vkd.createDescriptorSetLayout(gc.dev, .{
        .flags = .{},
        .binding_count = @truncate(u32, dslb.len),
        .p_bindings = @ptrCast([*]const vk.DescriptorSetLayoutBinding, &dslb),
    }, null);
}
fn createDescriptorSets(
    gc: GraphicsContext,
    allocator: *Allocator,
    descriptor_pool: vk.DescriptorPool,
    layout: vk.DescriptorSetLayout,
    unibufs: []const BufferMemory,
    texture: TextureImage,
) ![]vk.DescriptorSet {
    const size = @truncate(u32, unibufs.len);
    var layouts = try allocator.alloc(@TypeOf(layout), size);
    defer allocator.free(layouts);
    for (layouts) |*l| {
        l.* = layout;
    }
    const dsai = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = size,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, layouts),
    };
    var sets = try allocator.alloc(vk.DescriptorSet, size);
    try gc.vkd.allocateDescriptorSets(gc.dev, dsai, sets.ptr);

    for (unibufs) |unibuf, i| {
        const dbi = vk.DescriptorBufferInfo{
            .buffer = unibuf.buffer,
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };
        const dii = vk.DescriptorImageInfo{
            .sampler = texture.sampler,
            .image_view = texture.view,
            .image_layout = .shader_read_only_optimal,
        };
        const wds = [2]vk.WriteDescriptorSet{
            .{
                .dst_set = sets[i],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_image_info = undefined,
                .p_buffer_info = @ptrCast([*]const vk.DescriptorBufferInfo, &dbi),
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = sets[i],
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .combined_image_sampler,
                .p_image_info = @ptrCast([*]const vk.DescriptorImageInfo, &dii),
                .p_buffer_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        };
        gc.vkd.updateDescriptorSets(
            gc.dev,
            @truncate(u32, wds.len),
            @ptrCast([*]const vk.WriteDescriptorSet, &wds),
            0,
            undefined,
        );
    }
    return sets;
}
fn createUniformBuffer(gc: GraphicsContext, allocator: *Allocator, framebuffers: []const vk.Framebuffer) ![]BufferMemory {
    const unibufs = try allocator.alloc(BufferMemory, framebuffers.len);
    errdefer allocator.free(unibufs);

    const size = @sizeOf(UniformBufferObject);

    for (unibufs) |*buf| {
        buf.* = try BufferMemory.init(
            gc,
            size,
            .{ .uniform_buffer_bit = true },
            .{ .host_coherent_bit = true, .host_visible_bit = true },
        );
    }

    return unibufs;
}

fn destroyUniformBuffers(gc: GraphicsContext, allocator: *Allocator, bufs: []BufferMemory) void {
    for (bufs) |b| {
        b.deinit(gc);
    }
    allocator.free(bufs);
}

fn createDescriptorPool(gc: GraphicsContext, framebuffers: []const vk.Framebuffer) !vk.DescriptorPool {
    const size = @truncate(u32, framebuffers.len);
    var pool_size = [2]vk.DescriptorPoolSize{
        .{
            .@"type" = .uniform_buffer,
            .descriptor_count = size,
        },
        .{
            .@"type" = .combined_image_sampler,
            .descriptor_count = size,
        },
    };
    const dpci = vk.DescriptorPoolCreateInfo{
        .flags = .{},
        .max_sets = size,
        .pool_size_count = @truncate(u32, pool_size.len),
        .p_pool_sizes = @ptrCast([*]const vk.DescriptorPoolSize, &pool_size),
    };
    return try gc.vkd.createDescriptorPool(gc.dev, dpci, null);
}

pub fn createTextureImage(gc: GraphicsContext, pool: vk.CommandPool) !TextureImage {
    var tex_width: i32 = 0;
    var tex_height: i32 = 0;
    var tex_channels: i32 = 0;
    const pixels = c.stbi_load("assets/texture.jpg", &tex_width, &tex_height, &tex_channels, c.STBI_rgb_alpha);
    const size = @intCast(vk.DeviceSize, tex_width) * @intCast(vk.DeviceSize, tex_height) * 4;
    assert(pixels != null and size > 0);
    defer c.stbi_image_free(pixels);
    var stage_buffer = try BufferMemory.init(
        gc,
        size,
        .{ .transfer_src_bit = true },
        .{ .host_coherent_bit = true, .host_visible_bit = true },
    );
    defer stage_buffer.deinit(gc);

    {
        const data = try gc.vkd.mapMemory(gc.dev, stage_buffer.memory, 0, size, .{});
        defer gc.vkd.unmapMemory(gc.dev, stage_buffer.memory);

        const gpu_memory = @ptrCast([*]u8, @alignCast(@alignOf(u8), data));
        for (pixels[0..size]) |p, i| {
            gpu_memory[i] = p;
        }
    }
    const width = @intCast(u32, tex_width);
    const height = @intCast(u32, tex_height);
    const image = try TextureImage.init(
        gc,
        width,
        height,
        .r8g8b8a8_srgb,
        .optimal,
        .{ .transfer_dst_bit = true, .sampled_bit = true },
        .{ .device_local_bit = true },
    );
    //TODO continue Layout transitions and nor deinit image here
    try transitionImageLayout(gc, pool, image.image, .r8g8b8a8_srgb, .@"undefined", .transfer_dst_optimal);
    try copyBufferToImage(gc, pool, stage_buffer.buffer, image.image, width, height);
    try transitionImageLayout(
        gc,
        pool,
        image.image,
        .r8g8b8a8_srgb,
        .transfer_dst_optimal,
        .shader_read_only_optimal,
    );
    return image;
}

fn transitionImageLayout(
    gc: GraphicsContext,
    pool: vk.CommandPool,
    image: vk.Image,
    format: vk.Format,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
) !void {
    // TOTO: format
    _ = format;
    const TransferType = enum {
        @"undefined",
        transfer_dst_optimal,
        unsupported,
    };
    const cmdbuf = try beginSingleTimeCommand(gc, pool);
    const transfer_type: TransferType = blk: {
        if (old_layout == .@"undefined" and new_layout == .transfer_dst_optimal) break :blk .@"undefined";
        if (old_layout == .transfer_dst_optimal and new_layout == .shader_read_only_optimal) break :blk .transfer_dst_optimal;
        break :blk .unsupported;
    };
    assert(transfer_type != .unsupported);
    const imb = vk.ImageMemoryBarrier{
        .src_access_mask = if (transfer_type == .@"undefined") .{} else .{ .transfer_write_bit = true },
        .dst_access_mask = if (transfer_type == .@"undefined") .{ .transfer_write_bit = true } else .{ .shader_read_bit = true },
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    };
    gc.vkd.cmdPipelineBarrier(
        cmdbuf,
        if (transfer_type == .@"undefined") .{ .top_of_pipe_bit = true } else .{ .transfer_bit = true },
        if (transfer_type == .@"undefined") .{ .transfer_bit = true } else .{ .fragment_shader_bit = true },
        .{},
        0,
        undefined,
        0,
        undefined,
        1,
        @ptrCast([*]const vk.ImageMemoryBarrier, &imb),
    );
    try endSingleTimeCommands(gc, pool, cmdbuf);
}

fn copyBufferToImage(
    gc: GraphicsContext,
    pool: vk.CommandPool,
    buffer: vk.Buffer,
    image: vk.Image,
    width: u32,
    height: u32,
) !void {
    const cmdbuf = try beginSingleTimeCommand(gc, pool);

    const bic = vk.BufferImageCopy{
        .buffer_offset = 0,
        .buffer_row_length = 0,
        .buffer_image_height = 0,
        .image_subresource = .{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .base_array_layer = 0,
            .layer_count = 1,
        },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = width, .height = height, .depth = 1 },
    };
    gc.vkd.cmdCopyBufferToImage(
        cmdbuf,
        buffer,
        image,
        .transfer_dst_optimal,
        1,
        @ptrCast([*]const vk.BufferImageCopy, &bic),
    );
    try endSingleTimeCommands(gc, pool, cmdbuf);
}
fn beginSingleTimeCommand(
    gc: GraphicsContext,
    pool: vk.CommandPool,
) !vk.CommandBuffer {
    var cmdbuf: vk.CommandBuffer = undefined;
    try gc.vkd.allocateCommandBuffers(gc.dev, .{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast([*]vk.CommandBuffer, &cmdbuf));

    try gc.vkd.beginCommandBuffer(cmdbuf, .{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });
    return cmdbuf;
}

fn endSingleTimeCommands(gc: GraphicsContext, pool: vk.CommandPool, cmdbuf: vk.CommandBuffer) !void {
    defer gc.vkd.freeCommandBuffers(gc.dev, pool, 1, @ptrCast([*]const vk.CommandBuffer, &cmdbuf));
    try gc.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try gc.vkd.queueSubmit(gc.graphics_queue.handle, 1, @ptrCast([*]const vk.SubmitInfo, &si), .null_handle);
    try gc.vkd.queueWaitIdle(gc.graphics_queue.handle);
}
fn createDepthResources(gc: GraphicsContext, extent: vk.Extent2D) !DepthImage {
    const depth_format = DepthImage.findDepthFormat(gc).?;
    const depth_image = try DepthImage.init(
        gc,
        extent.width,
        extent.height,
        depth_format,
        .optimal,
        .{ .depth_stencil_attachment_bit = true },
        .{ .device_local_bit = true },
    );
    return depth_image;
    // TODO: Explicitly transitioning the depth image
    // try transitionImageLayout(gc, pool, image.image, .r8g8b8a8_srgb, .@"undefined", .transfer_dst_optimal);
}
