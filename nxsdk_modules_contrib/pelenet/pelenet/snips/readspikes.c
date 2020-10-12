#include <stdlib.h>
#include <string.h>
#include "runmgmt.h"
#include <time.h>
#include <unistd.h>

// Initialize spike array
int spike = 0;
int channelID = -1;

int do_run_mgmt(runState *s) {
    if (s->time_step==1){
        channelID = getChannelID("nxspkcntr");
    }
    return 1;
}

void run_mgmt(runState *s) {
    // In every iteration (time step) send spike information for every neuron through channel
    for (int i = 0; i < 4500; i++) {
        // Get spike for specific neuron (from specific probe ID)
        spike = SPIKE_COUNT[(s->time_step-1)&3][0x20+i];
        // Reset spike value
        SPIKE_COUNT[(s->time_step-1)&3][0x20+i] = 0;
        // Write spike for current neuron to channel
        writeChannel(channelID, &spike, 1);
    }
}
