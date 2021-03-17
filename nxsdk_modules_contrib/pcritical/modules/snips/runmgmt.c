#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "runmgmt.h"
#include "constants.h"

uint8_t spikeCounts[nb_of_neurons] = {0};
_Static_assert(bin_size < 256,  "We want the bins to fit in a uint8 (also before SPIKE_COUNT overflows)");

int do_run_mgmt(runState *s) {
    return s->time_step % bin_size == 0; // time_step starts at 1
}


void run_mgmt(runState *s) {
    static int channelID = -1;
    if (channelID == -1) {
        channelID = getChannelID("nxspkcntr");
    }

    for (uint32_t i = 0; i < nb_of_neurons; ++i) {
        for(int j=0; j<4; j++) { // TODO: Validate necessity
            spikeCounts[i] += SPIKE_COUNT[j][i+0x20];
            SPIKE_COUNT[j][i+0x20] = 0;
        }
    }
    //printf("Writing to channel at time step : %d\n",s->time_step);
    writeChannel(channelID, spikeCounts, 1);
    memset(spikeCounts, 0, sizeof spikeCounts);
}
