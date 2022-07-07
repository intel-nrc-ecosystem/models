/*
INTEL CONFIDENTIAL

Copyright © 2021 Intel Corporation.

This software and the related documents are Intel copyrighted
materials, and your use of them is governed by the express
license under which they were provided to you (License). Unless
the License provides otherwise, you may not use, modify, copy,
publish, distribute, disclose or transmit  this software or the
related documents without Intel's prior written permission.

This software and the related documents are provided as is, with
no express or implied warranties, other than those that are
expressly stated in the License.
*/

#include <iostream>
#include <fstream>
#include <unistd.h>
#include "nxsdkhost.h"
// #include "load_register.h" // LK: Removed for debug
#include "host_snip_load_reg.h"
// #include <time.h>
# include <chrono>

// LK: added for debug
namespace nx{
    void loadBoardStateGivenChipId(std::string filename_a, uint32_t preassignedChipId);
}

class LoadChips : public PreExecutionSequentialHostSnip 
{
    private:
        std::string channel = "load_chips";
    public:
        virtual void run(uint32_t timestep) 
        {
            // Send data except for the last timestep where you send 0
            // clock_t start, end;
            // start = clock();
            auto start = std::chrono::high_resolution_clock::now();

            for(uint32_t chipId = 0; chipId < NUM_CHIPS; chipId++) 
            {
                if (access(FILE_NAME, F_OK) != -1) 
                {            
                    std::cout << "Reading filename: " << FILE_NAME << " Programming Chip: " << chipId << std::endl;
                    nx::loadBoardStateGivenChipId(FILE_NAME, chipId);
                } 
                else 
                {
                    std::cout << "Filename: " << FILE_NAME << " does not exist!!! " << std::endl;
                    throw 10;
                }
            }
            // end = clock();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Time Taken: " << duration.count()/1000.0 << " seconds." << std::endl;
        }

        virtual std::valarray<uint32_t> schedule(const std::valarray<uint32_t>& timesteps) const 
        {
            // Only execute on first timestep when the network needs to be programmed
            return {1};
        }
};

REGISTER_SNIP(LoadChips, PreExecutionSequentialHostSnip);
