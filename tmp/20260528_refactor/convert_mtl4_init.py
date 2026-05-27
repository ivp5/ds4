#!/usr/bin/env python3
"""Refactor MTL4 pipeline-init functions to use ds4_mtl4_build_kernel_pipeline.

Scans ds4_metal.m for the boilerplate pattern:
  MTL4LibraryDescriptor *libDesc = [MTL4LibraryDescriptor new];
  libDesc.source = source;
  libDesc.name = @"LIB_NAME";
  id<MTLLibrary> lib = [g_polar_compiler newLibraryWithDescriptor:libDesc error:&err];
  if (!lib) {
      fprintf(stderr, "ds4: KERNEL_NAME MTL4 library compile failed: %s\n",
              err ? err.localizedDescription.UTF8String : "(no error)");
      return 0;
  }
  MTL4LibraryFunctionDescriptor *fnDesc = [MTL4LibraryFunctionDescriptor new];
  fnDesc.library = lib;
  fnDesc.name = @"KERNEL_NAME";
  MTL4ComputePipelineDescriptor *pipeDesc = [MTL4ComputePipelineDescriptor new];
  pipeDesc.computeFunctionDescriptor = fnDesc;
  pipeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
  pipeDesc.maxTotalThreadsPerThreadgroup = MAX;
  g_PIPELINE =
      [g_polar_compiler newComputePipelineStateWithDescriptor:pipeDesc
                                         compilerTaskOptions:nil error:&err];
  if (!g_PIPELINE) {
      fprintf(stderr, "ds4: KERNEL_NAME MTL4 pipeline failed: %s\n",
              err ? err.localizedDescription.UTF8String : "(no error)");
      return 0;
  }
  fprintf(stderr, "ds4: KERNEL_NAME MTL4 pipeline initialized\n");
  g_INIT_OK = 1;
  return 1;

And replaces with:
  g_PIPELINE = ds4_mtl4_build_kernel_pipeline(
      source, @"LIB_NAME", @"KERNEL_NAME", MAX, NULL, 0);
  g_INIT_OK = (g_PIPELINE != nil) ? 1 : 0;
  return g_INIT_OK;

Also removes preceding:
  if (!ds4_polar_pipeline_init()) return 0;
  NSError *err = nil;

Run from project root. Reports how many sites converted.
"""
import re
import sys
from pathlib import Path

PATH = Path("/Users/silv/cl/tlp/montyneg/ivp5_ds4/ds4_metal.m")
src = PATH.read_text()

# Pattern for the full boilerplate block. Captures lib_name, kernel_name (must match
# in compile-failed + pipeline-failed + initialized log), max_threads, pipeline_var,
# init_ok_var.
PATTERN = re.compile(
    r'    MTL4LibraryDescriptor \*libDesc = \[MTL4LibraryDescriptor new\];\n'
    r'    libDesc\.source = source;\n'
    r'    libDesc\.name = @"(?P<lib>[^"]+)";\n'
    r'    id<MTLLibrary> lib = \[g_polar_compiler newLibraryWithDescriptor:libDesc error:&err\];\n'
    r'    if \(!lib\) \{\n'
    r'        fprintf\(stderr, "ds4: (?P<kfail1>[^"]+) MTL4 library compile failed: %s\\n",\n'
    r'                err \? err\.localizedDescription\.UTF8String : "\(no error\)"\);\n'
    r'        return 0;\n'
    r'    \}\n'
    r'    MTL4LibraryFunctionDescriptor \*fnDesc = \[MTL4LibraryFunctionDescriptor new\];\n'
    r'    fnDesc\.library = lib;\n'
    r'    fnDesc\.name = @"(?P<kernel>[^"]+)";\n'
    r'    MTL4ComputePipelineDescriptor \*pipeDesc = \[MTL4ComputePipelineDescriptor new\];\n'
    r'    pipeDesc\.computeFunctionDescriptor = fnDesc;\n'
    r'    pipeDesc\.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;\n'
    r'    pipeDesc\.maxTotalThreadsPerThreadgroup = (?P<max>\d+);\n'
    r'    (?P<pipevar>g_[a-zA-Z0-9_]+) =\n'
    r'        \[g_polar_compiler newComputePipelineStateWithDescriptor:pipeDesc\n'
    r'                                           compilerTaskOptions:nil error:&err\];\n'
    r'    if \(!(?P=pipevar)\) \{\n'
    r'        fprintf\(stderr, "ds4: (?P<kfail2>[^"]+) MTL4 pipeline failed: %s\\n",\n'
    r'                err \? err\.localizedDescription\.UTF8String : "\(no error\)"\);\n'
    r'        return 0;\n'
    r'    \}\n'
    r'    fprintf\(stderr, "ds4: (?P<klog>[^"]+) MTL4 pipeline initialized\\n"\);\n'
    r'    (?P<okvar>g_[a-zA-Z0-9_]+_init_ok) = 1;\n'
    r'    return 1;\n'
)

# Also match the preceding "if polar init / NSError *err = nil;" pair
PRE_PATTERN = re.compile(
    r'    if \(!ds4_polar_pipeline_init\(\)\) return 0;\n'
    r'\n'
    r'    NSError \*err = nil;\n'
    r'    NSString \*source =\n',
)

count = 0
def replace(m):
    global count
    count += 1
    lib = m.group("lib")
    kernel = m.group("kernel")
    max_t = m.group("max")
    pipe = m.group("pipevar")
    ok = m.group("okvar")
    return (
        f'    {pipe} = ds4_mtl4_build_kernel_pipeline(\n'
        f'        source, @"{lib}", @"{kernel}", {max_t}, NULL, 0);\n'
        f'    {ok} = ({pipe} != nil) ? 1 : 0;\n'
        f'    return {ok};\n'
    )

new_src = PATTERN.sub(replace, src)
# Also replace the pre-block: "if (!ds4_polar_pipeline_init()) return 0;\n\n    NSError *err = nil;\n    NSString *source =\n"
# with "    NSString *source =\n"
new_src = PRE_PATTERN.sub('    NSString *source =\n', new_src)

print(f"converted {count} pipeline-init sites", file=sys.stderr)

if count > 0:
    PATH.write_text(new_src)
    print(f"  saved approximately {count * 25} lines", file=sys.stderr)
