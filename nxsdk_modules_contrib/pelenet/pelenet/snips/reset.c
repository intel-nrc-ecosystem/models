#include <stdlib.h>
#include <string.h>
#include "time.h"
#include "reset.h"

static int numCores = 128;

extern int neuronsPerCore;
extern int resetInterval;
extern int resetSteps;

int do_reset(runState *RunState) {
    bool apply = false;

    // Prepare boolean variable, which is true for 10 time steps
    for(int i = 1; i <= resetSteps; i++) {
        apply = apply || (RunState->time_step - i) % resetInterval == 0;
    }

    // Return boolean for guard for reset() function
    if (RunState->time_step == 1 || apply) {
        return 1;
    } else {
        return 0;
    }
}

void reset(runState *RunState) {

    // Define new voltage and current states (voltage 0, current 0)
    CxState cxs = (CxState) {.U=0, .V=0};

    // Define new functional state IDLE of compartments
    MetaState ms = (MetaState) {.Phase0=2, .SomaOp0=3,
                                .Phase1=2, .SomaOp1=3,
                                .Phase2=2, .SomaOp2=3,
                                .Phase3=2, .SomaOp3=3};

    // Iterate over all cores
    for (int i = 0; i < numCores; i++) {
        // Get core of current iteration
        NeuronCore* nc = NEURON_PTR(nx_nth_coreid(i));

        // Sets all 1024 registers on the neurocore to new current and voltage
        nx_fast_init64(nc->cx_state, neuronsPerCore, *(uint64_t*)&cxs);

        // Flush core
        nx_flush_core(nx_nth_coreid(i));
    }

    // Only reset functional state in last stop step
    if ((RunState->time_step - resetSteps) % resetInterval == 0) {
        // Print what we are doing
        printf("Reset current/voltage from time step %d to %d\n", RunState->time_step-resetSteps, RunState->time_step);
        printf("Reset functional state at %d\n", RunState->time_step);

        // Set functional state for all cores
        for(int i=0; i<numCores; i++) {
            NeuronCore* nc = NEURON_PTR(nx_nth_coreid(i));
            nx_fast_init32(nc->cx_meta_state, neuronsPerCore/4, *(uint32_t*)&ms);
            //nx_flush_core(nx_nth_coreid(i)); // ?
        }
    }
}


