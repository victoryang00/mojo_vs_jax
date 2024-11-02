/* Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string>

#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_buffer.h"
// #include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
// #include "tsl/platform/test.h"

namespace xla
{
    namespace xla_compile
    {
        namespace
        {

            int main(int argc, char *argv[])
            {
                std::string path =
                    "./mlp.o";
                std::string serialized_aot_result;

                tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result);

                // Get a LocalClient
                se::Platform *platform,
                    PlatformUtil::GetPlatform("Host");
                if (platform->VisibleDeviceCount() <= 0)
                {
                    EXPECT_TRUE(false) << "CPU platform has no visible devices.";
                }
                LocalClientOptions local_client_options;
                local_client_options.set_platform(platform);

                LocalClient *client,
                    ClientLibrary::GetOrCreateLocalClient(local_client_options);

                // Load from AOT result.
                ExecutableBuildOptions executable_build_options;
                //   executable_build_options =
                //   executable_build_options.set_exec_time_optimization_effort(100);

                std::unique_ptr<LocalExecutable> local_executable,
                    client->Load(serialized_aot_result, executable_build_options);

                // Run loaded excutable.
                Literal input2 = LiteralUtil::CreateR1<float>(
                    {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                     1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                     1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                     1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                     1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f,
                     1.0f, 2.0f, 0.0f, 1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f});
                //   std::vector<std::vector<float>> input_data(128,
                //   std::vector<float>(input2));

                // Create a Literal from the 2D vector.
                //  input2 =
                //  LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(input_data));
                xla::Array2D<float> input_array(128, 128);
                for (int i = 0; i < 128; ++i)
                {
                    for (int j = 0; j < 128; ++j)
                    {
                        input_array(i, j) = 1;
                    }
                }
                Literal input1 = LiteralUtil::CreateR2FromArray2D<float>(input_array);
                xla::Array2D<float> input_array1(1, 128);
                for (int i = 0; i < 1; ++i)
                {
                    for (int j = 0; j < 128; ++j)
                    {
                        input_array(i, j) = 1;
                    }
                }
                Literal input3 = LiteralUtil::CreateR2FromArray2D<float>(input_array1);
                //   Literal input2 = LiteralUtil::CreateR1<double>({1.0f, 2.0f, 4.0f});

                ScopedShapedBuffer array1,
                    client->LiteralToShapedBuffer(input1, client->default_device_ordinal());

                ScopedShapedBuffer array2,
                    client->LiteralToShapedBuffer(input2, client->default_device_ordinal());

                ScopedShapedBuffer array3,
                    client->LiteralToShapedBuffer(input3, client->default_device_ordinal());
                ExecutableRunOptions executable_run_options;
                executable_run_options.set_allocator(client->backend().memory_allocator());
                int time = clock();
                for (int i = 0; i < 1000; i++)
                {
                    absl::StatusOr<xla::ScopedShapedBuffer> result = local_executable->Run(
                        absl::Span<const ShapedBuffer *const>{
                            &array1, &array2, &array1, &array2, &array1, &array2, &array1,
                            &array2, &array1, &array2, &array1, &array2, &array1, &array2,
                            &array1, &array2, &array3},
                        executable_run_options);
                }
                printf("Time taken: %ld\n", clock() - time);
            }

        } // namespace
    } // namespace xla_compile
} // namespace xla
