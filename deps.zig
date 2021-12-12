const std = @import("std");
const builtin = @import("builtin");
const Pkg = std.build.Pkg;
const string = []const u8;

pub const cache = ".zigmod\\deps";

pub fn addAllTo(exe: *std.build.LibExeObjStep) void {
    @setEvalBranchQuota(1_000_000);
    for (packages) |pkg| {
        exe.addPackage(pkg.pkg.?);
    }
    var llc = false;
    var vcpkg = false;
    inline for (std.meta.declarations(package_data)) |decl| {
        const pkg = @as(Package, @field(package_data, decl.name));
        inline for (pkg.system_libs) |item| {
            exe.linkSystemLibrary(item);
            llc = true;
        }
        inline for (pkg.c_include_dirs) |item| {
            exe.addIncludeDir(@field(dirs, decl.name) ++ "/" ++ item);
            llc = true;
        }
        inline for (pkg.c_source_files) |item| {
            exe.addCSourceFile(@field(dirs, decl.name) ++ "/" ++ item, pkg.c_source_flags);
            llc = true;
        }
    }
    if (llc) exe.linkLibC();
    if (builtin.os.tag == .windows and vcpkg) exe.addVcpkgPaths(.static) catch |err| @panic(@errorName(err));
}

pub const Package = struct {
    directory: string,
    pkg: ?Pkg = null,
    c_include_dirs: []const string = &.{},
    c_source_files: []const string = &.{},
    c_source_flags: []const string = &.{},
    system_libs: []const string = &.{},
    vcpkg: bool = false,
};

const dirs = struct {
    pub const _root = "";
    pub const _3oldm2uf7rpx = cache ++ "/../..";
    pub const _uxw7q1ovyv4z = cache ++ "/git/github.com/Snektron/vulkan-zig";
};

pub const package_data = struct {
    pub const _3oldm2uf7rpx = Package{
        .directory = dirs._3oldm2uf7rpx,
    };
    pub const _uxw7q1ovyv4z = Package{
        .directory = dirs._uxw7q1ovyv4z,
        .pkg = Pkg{ .name = "vulkan-zig", .path = .{ .path = dirs._uxw7q1ovyv4z ++ "/generator/index.zig" }, .dependencies = null },
    };
    pub const _root = Package{
        .directory = dirs._root,
    };
};

pub const packages = &[_]Package{
    package_data._uxw7q1ovyv4z,
};

pub const pkgs = struct {
    pub const vulkan_zig = package_data._uxw7q1ovyv4z;
};

pub const imports = struct {
    pub const vulkan_zig = @import(".zigmod\\deps/git/github.com/Snektron/vulkan-zig/generator/index.zig");
};
