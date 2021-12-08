const std = @import("std");
const deps = @import("./deps.zig");
const vkgen = deps.imports.vulkan_zig;
const Step = std.build.Step;
const Builder = std.build.Builder;
const LibExeObjStep = std.build.LibExeObjStep;
const Pkg = std.build.Pkg;

pub const ResourceGenStep = struct {
    step: Step,
    shader_step: *vkgen.ShaderCompileStep,
    builder: *Builder,
    package: std.build.Pkg,
    output_file: std.build.GeneratedFile,
    resources: std.ArrayList(u8),

    pub fn init(builder: *Builder, out: []const u8) *ResourceGenStep {
        const self = builder.allocator.create(ResourceGenStep) catch unreachable;
        const full_out_path = std.fs.path.join(builder.allocator, &[_][]const u8{
            builder.build_root,
            builder.cache_root,
            out,
        }) catch unreachable;

        self.* = .{
            .step = Step.init(.custom, "resources", builder.allocator, make),
            .shader_step = vkgen.ShaderCompileStep.init(builder, &[_][]const u8{ "glslc", "--target-env=vulkan1.1" }),
            .builder = builder,
            .package = .{
                .name = "resources",
                .path = .{ .generated = &self.output_file },
                .dependencies = null,
            },
            .output_file = .{
                .step = &self.step,
                .path = full_out_path,
            },
            .resources = std.ArrayList(u8).init(builder.allocator),
        };

        self.step.dependOn(&self.shader_step.step);
        return self;
    }

    fn renderPath(path: []const u8, writer: anytype) void {
        const separators = &[_]u8{ std.fs.path.sep_windows, std.fs.path.sep_posix };
        var i: usize = 0;
        while (std.mem.indexOfAnyPos(u8, path, i, separators)) |j| {
            writer.writeAll(path[i..j]) catch unreachable;
            switch (std.fs.path.sep) {
                std.fs.path.sep_windows => writer.writeAll("\\\\") catch unreachable,
                std.fs.path.sep_posix => writer.writeByte(std.fs.path.sep_posix) catch unreachable,
                else => unreachable,
            }

            i = j + 1;
        }
        writer.writeAll(path[i..]) catch unreachable;
    }

    pub fn addShader(self: *ResourceGenStep, name: []const u8, source: []const u8) void {
        const shader_out_path = self.shader_step.add(source);
        var writer = self.resources.writer();

        writer.print("pub const {s} = @embedFile(\"", .{name}) catch unreachable;
        renderPath(shader_out_path, writer);
        writer.writeAll("\");\n") catch unreachable;
    }

    fn make(step: *Step) !void {
        const self = @fieldParentPtr(ResourceGenStep, "step", step);
        const cwd = std.fs.cwd();

        const dir = std.fs.path.dirname(self.output_file.path.?).?;
        try cwd.makePath(dir);
        try cwd.writeFile(self.output_file.path.?, self.resources.items);
    }
};
pub fn linkGlfw(b: *LibExeObjStep) void {
    const glfw = b.builder.addStaticLibrary("glfw", null);
    const glfw_path = "./external/glfw/";
    inline for (.{
        "win32_thread.c",
        "wgl_context.c",
        "win32_init.c",
        "win32_monitor.c",
        "win32_time.c",
        "win32_joystick.c",
        "win32_window.c",
        "win32_module.c",

        "null_init.c",
        "null_monitor.c",
        "null_window.c",
        "null_joystick.c",
        "platform.c",
        "monitor.c",
        "init.c",
        "vulkan.c",
        "input.c",
        "osmesa_context.c",
        "egl_context.c",
        "context.c",
        "window.c",
    }) |f| {
        glfw.addCSourceFile(glfw_path ++ "src/" ++ f, &.{"-D_GLFW_WIN32"});
    }
    glfw.linkLibC();
    glfw.addIncludeDir(glfw_path ++ "src");
    b.addIncludeDir(glfw_path ++ "include");
    b.linkLibrary(glfw);
}

pub fn linkStbImage(b: *LibExeObjStep) void {
    const stb_image = b.builder.addStaticLibrary("stb_image", null);
    stb_image.addCSourceFile("external/stb_image.c", &.{});
    stb_image.linkLibC();
    b.addIncludeDir("external");
    b.linkLibrary(stb_image);
}

pub fn build(b: *Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const triangle_exe = b.addExecutable("triangle", "./src/triangle.zig");
    triangle_exe.setTarget(target);
    triangle_exe.setBuildMode(mode);
    triangle_exe.install();
    triangle_exe.linkLibC();
    triangle_exe.linkSystemLibrary("gdi32");
    linkGlfw(triangle_exe);
    linkStbImage(triangle_exe);

    const vk_sdk_path = b.option([]const u8, "vulkan-sdk", "Path to vulkan sdk");
    const gen = if (vk_sdk_path) |path|
        vkgen.VkGenerateStep.initFromSdk(b, path, "vk.zig").package
    else
        Pkg{ .name = "vulkan", .path = .{ .path = "zig-cache/vk.zig" }, .dependencies = null };

    triangle_exe.addPackage(gen);

    const compile_shader = b.option(bool, "shader", "Compile shader when build");
    const res = blk: {
        if (compile_shader) |_| {
            const res = ResourceGenStep.init(b, "resources.zig");
            res.addShader("triangle_vert", "src/shaders/triangle.vert");
            res.addShader("triangle_frag", "src/shaders/triangle.frag");
            break :blk res.package;
        }

        break :blk Pkg{ .name = "resources", .path = .{ .path = "zig-cache/resources.zig" }, .dependencies = null };
    };

    triangle_exe.addPackage(res);
    const zalgebra = Pkg{ .name = "zalgebra", .path = .{ .path = "zalgebra/src/main.zig" }, .dependencies = null };
    triangle_exe.addPackage(zalgebra);

    const triangle_run_cmd = triangle_exe.run();
    triangle_run_cmd.step.dependOn(b.getInstallStep());
    const triangle_run_step = b.step("run", "Run the triangle example");
    triangle_run_step.dependOn(&triangle_run_cmd.step);
}
