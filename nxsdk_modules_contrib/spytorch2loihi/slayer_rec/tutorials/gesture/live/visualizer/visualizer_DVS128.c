/*
INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY

Copyright © 2019-2021 Intel Corporation.

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

// A SpikeOutputPort in NxSDK encodes a portId, when a spike is sent to this
// port, the spike is transported to the host, and the corresponding portId can
// be read from a named pipe specified by the NX_SPIKE_OUTPUT environment
// variable.
//
// This program demonstrates this interface by visualizing DVS spikes sent to
// the host, assuming the portId encoding is compatible with DVSSpikeGen.
//
// Compile with:
// gcc -O3 $(sdl2-config --cflags) visualize_kb_spikes.c $(sdl2-config --libs) -o visualize_kb_spikes
//
// Run on KB with:
// mkfifo /path/to/pipe
// visualize_kb_spikes --source=/path/to/pipe &
// NX_SPIKE_OUTPUT=/path/to/pipe KAPOHOBAY=1 python3 -m nxsdk.nxsdk_modules.noise_filter.tutorials.dvs_with_noise_filter
//
// Example for the dvs half filter demo:
// from the root of your repo (nxsdk-nxsdk)
// > gcc -O3 $(sdl2-config --cflags) nxsdk_modules/noise_filter/src/visualize_kb_spikes.c $(sdl2-config --libs) -o nxsdk_modules/noise_filter/src/visualize_kb_spikes
// > NX_SPIKE_OUTPUT=nxsdk_modules/noise_filter/spikefifo
// > mkfifo $NX_SPIKE_OUTPUT
// > ./nxsdk_modules/noise_filter/src/visualize_kb_spikes --source=$NX_SPIKE_OUTPUT &  KAPOHOBAY=1 NX_SPIKE_OUTPUT=$NX_SPIKE_OUTPUT python -m nxsdk_modules.noise_filter.tutorials.dvs_with_noise_filter

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include "SDL.h"

#define DVS_X 128
#define DVS_Y 128
#define DVS_P 2

#define USE_SDL
#define SDL_SCALE 4
#define SDL_INTERVAL_USEC 100000

typedef struct {
  uint32_t spikes;
} stats;

typedef struct __attribute__((packed)) {
  uint32_t portId:25,
                 : 7;
} spike;

#define CLOCK_T struct timeval 
#define GET_CLOCK(t) gettimeofday(&t, 0)
#define DIFF_CLOCK(tsub, t0, t1)  timersub(&t1, &t0, &tsub)
#define CLOCK_SEC(t) (t.tv_sec)
#define SEC_CLOCK(s) ((CLOCK_T) { .tv_sec = s, .tv_usec = 0 })
#define CLOCK_LESS(t0, t1) (timercmp(&t0, &t1, <))

#ifdef USE_SDL
SDL_Window *window;
SDL_Renderer *renderer;
#endif

void visualize(int fd, int max, CLOCK_T interval, int verbose) {
  struct pollfd fds = { fd, POLLIN | POLLOUT, 0 };

  stats s = {0};
  CLOCK_T start;
  GET_CLOCK(start);
  CLOCK_T sdlStart;
  GET_CLOCK(sdlStart);

#ifdef USE_SDL
  CLOCK_T sdlInterval = (CLOCK_T) { .tv_sec = 0, .tv_usec = SDL_INTERVAL_USEC };
  SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
  SDL_RenderClear(renderer);
#endif

  for (int i = 0; max == 0 || i < max; ) {
#ifdef USE_SDL
    SDL_Event event;
    SDL_PollEvent(&event);
    if (event.type == SDL_QUIT) {
      break;
    }
#endif

    fds.revents = 0;
    int avail = poll(&fds, 1, 0);
    if (avail == -1 || (fds.revents & (POLLERR | POLLNVAL))) {
      printf("Error reading spike input: %d %x\n", avail, fds.revents);
      exit(1);
    } else if (fds.revents & POLLHUP) {
      break;
    }
    if (avail) {
      spike spk;
      int r = read(fd, &spk, sizeof(spk));
      if (r == 0) {
        break;
      } else if (r != sizeof(spk)) {
        printf("Error reading spike input: %d %d\n", avail, fds.revents);
        exit(1);
      }
      uint32_t idx = spk.portId;
      uint32_t x = idx / (DVS_Y * DVS_P);
      uint32_t yp = idx % (DVS_Y * DVS_P);
      uint32_t y = 127-(yp / DVS_P);
      uint32_t p = yp % DVS_P;
      s.spikes++;
      if (verbose) {
        printf("spike %d: portId=%x x=%d y=%d p=%d\n", s.spikes, idx, x, y, p);
      }
#ifdef USE_SDL
      SDL_SetRenderDrawColor(renderer, p ? 255 : 0, p ? 0 : 255, 0x00, SDL_ALPHA_OPAQUE);
      SDL_RenderDrawPoint(renderer, x, (DVS_Y - 1 - y));
#endif
    }

    CLOCK_T curr;
    GET_CLOCK(curr);

#ifdef USE_SDL
    CLOCK_T sdlDiff;
    DIFF_CLOCK(sdlDiff, sdlStart, curr);
    if (CLOCK_LESS(sdlInterval, sdlDiff)) {
      SDL_RenderPresent(renderer);
      SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);
      SDL_RenderClear(renderer);
      GET_CLOCK(sdlStart);
    }
#endif

    CLOCK_T diff;
    DIFF_CLOCK(diff, start, curr);
    if (CLOCK_LESS(interval, diff)) {
      unsigned int t = CLOCK_SEC(diff);
      //if (t > 0) {
      //  printf("%u spikes %u spikes/s\n", s.spikes, s.spikes / t);
      //}
      s = (stats) {0};
      i++;

      GET_CLOCK(start);
    }
  }
}

int main(int argc, char **argv) {
  unsigned int max = 0, verbose = 0, interval = 10;
  char filename[128] = {0};
  for (int i = 1; i < argc; i++) {
    if      (sscanf(argv[i], "--max=%u", &max) == 1);
    else if (sscanf(argv[i], "--verbose=%u", &verbose) ==1);
    else if (sscanf(argv[i], "--interval=%u", &interval) ==1);
    else if (sscanf(argv[i], "--source=%127s", filename) ==1);
    else    { printf("ERROR: unrecognized argument %s\n", argv[i]); exit(1); }
  }

  setbuf(stdout, 0);
  setbuf(stderr, 0);

  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    printf("Unable to open spike input\n");
    return 1;
  }

#ifdef USE_SDL
  if (SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
    return 1;
  }

  if (SDL_CreateWindowAndRenderer(DVS_X * SDL_SCALE, DVS_Y * SDL_SCALE, SDL_WINDOW_RESIZABLE, &window, &renderer)) {
    SDL_Log("Couldn't create window and renderer: %s", SDL_GetError());
    return 1;
  }
  if (SDL_RenderSetScale(renderer, SDL_SCALE, SDL_SCALE)) {
    SDL_Log("Couldn't set renderer scale: %s", SDL_GetError());
  }
#endif

  CLOCK_T clockval = SEC_CLOCK(interval);
  visualize(fd, max, clockval, verbose);
  close(fd);
  printf("Closing Visualizer\n");
  return 0;
}
