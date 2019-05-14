#include <SDL2/SDL.h>
#include <assert.h>
#include <iostream>

#include "camera.hh"
#include "mapfile.hh"
#include "topology.hh"
#include "multitron.hh"
#include "numutils.hh"
#include "imgutils.hh"
#include "strutils.hh"
#include "cudamem.hh"

using namespace makemore;

static bool *sampused;
static double *samprgb;
static double *samptgt;

int main( int argc, char* argv[] )
{
 Camera cam;
  cam.open();

  Topology *segtop;
  Mapfile *segmap;
  Multitron *seg;
{
  char segmapfn[4096], segtopfn[4096];
  sprintf(segtopfn, "%s/seg.top", "oldseg.proj");
  sprintf(segmapfn, "%s/seg.map", "oldseg.proj");
  segtop = new Topology;
  segtop->load_file(segtopfn);
  segmap = new Mapfile(segmapfn);
  seg = new Multitron(*segtop, segmap, 1, false);
}

    int posX = 0;
    int posY = 0;
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_DisplayMode DM;
    unsigned int w, h;
    int sizeX, sizeY;

    // ==========================================================
    if ( SDL_Init( SDL_INIT_EVERYTHING ) != 0 )
    {
        // Something failed, print error and exit.
        std::cout << " Failed to initialize SDL : " << SDL_GetError() << std::endl;
        return -1;
    }

    assert(0 == SDL_GetCurrentDisplayMode(0, &DM));
    w = DM.w;
    h = DM.h;
    sizeX = w;
    sizeY = h;

fprintf(stderr, "w=%u h=%u\n", w, h);
 
    // Initialize SDL
    // Create and init the window
    // ==========================================================
    window = SDL_CreateWindow( "Server", posX, posY, sizeX, sizeY, 0 );

    if ( window == nullptr )
    {
        std::cout << "Failed to create window : " << SDL_GetError();
        return -1;
    }

SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);

    // Create and init the renderer
    // ==========================================================
    renderer = SDL_CreateRenderer( window, -1, SDL_RENDERER_ACCELERATED );

    if ( renderer == nullptr )
    {
        std::cout << "Failed to create renderer : " << SDL_GetError();
        return -1;
    }

    // Render something
    // ==========================================================

    // Set size of renderer to the same as window
    SDL_RenderSetLogicalSize( renderer, sizeX, sizeY );
     
    SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );
    SDL_RenderClear( renderer );
    SDL_RenderPresent( renderer);

int nw = 1280/2, nh = 720/2;
  unsigned int nsamp = 100;
  samprgb = new double[nw * nh * 3 * nsamp];
  sampused = new bool[nsamp]();
  samptgt = new double[2 * nsamp];
  unsigned int isamp = 0;

uint8_t *rgb0 = new uint8_t[nw * nh * 3]();
uint8_t *rgb1 = new uint8_t[nw * nh * 3]();
uint8_t *rgb = new uint8_t[nw * nh * 3];
double *drgb = new double[nw * nh * 3];
double *drgb2 = new double[nw * nh * 3];
double *segin;
cumake(&segin, 256 * 256 * 3);

double tgtx = 0.5, tgty = 0.5;
double *segtgt;
cumake(&segtgt, 2);
bool tdown = 0;
int frame = 0;

if (argc > 1) frame = atoi(argv[1]);
double xy[6];
xy[0] = nw/2;
xy[1] = nh/2;
xy[2] = nw/2;
xy[3] = nh/2;
xy[4] = nw/2;
xy[5] = nh/2;

double s = 3.0;

while (1) {

fprintf(stderr, "tdown=%d\n", tdown);


SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );
SDL_RenderClear( renderer );


memcpy(rgb0, rgb1, sizeof(uint8_t) * nw * nh * 3);
    cam.read(rgb1, nw, nh, true);

for (unsigned int j = 0; j < nw * nh * 3; ++j)
  rgb[j] = rgb1[j];
#if 0
for (unsigned int j = 0; j < nw * nh * 3; ++j)
  rgb[j] = abs(rgb1[j] - rgb0[j]);
double ss = 0;
for (unsigned int j = 0; j < nw * nh * 3; j += 3) {
  double m = rgb[j];
  if (rgb[j+1] > m) m = rgb[j+1];
  if (rgb[j+2] > m) m = rgb[j+2];
m *= 2;
  rgb[j] = m;
  rgb[j+1] = m;
  rgb[j+2] = m;
  ss += m;
}
ss /= nw *nh*3;
fprintf(stderr, "ss=%lf\n", ss);
#endif

++frame;
if (tdown) {
  std::string png;
  rgbpng(rgb, nw, nh, &png);
  char fn[256];
  sprintf(fn, "cam/camera.%08d.png", frame);
  spit(png, fn);
}

SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(rgb, nw, nh, 24, nw * 3, 0xff, 0xff00, 0xff0000, 0);
SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, surf);

SDL_Rect srect;
srect.x = 0;
srect.y = 0;
srect.w = nw;
srect.h = nh;
SDL_Rect drect;
drect.x = w/2-nw*s/2.0;
drect.y = h/2-nh*s/2.0;
drect.w = s*nw;
drect.h = s*nh;
SDL_RenderCopy(renderer, text, &srect, &drect);


for (unsigned int k = 0; k < 3; ++k) {
btodv(rgb, drgb, nw * nh * 3);

int offx = (nw - 256)/2;
int offy = (nh - 256)/2;

double cx = xy[0]/4.0 + xy[2]/4.0 + xy[4]/2.0;
double cy = xy[1]/4.0 + xy[3]/4.0 + xy[5]/2.0;

offx = cx - 128.0;
offy = cy - 128.0;

double *qq = drgb2;
for (int y = offy; y < offy + 256; ++y) {
  for (int x = offx; x < offx + 256; ++x) {
    for (int c = 0; c < 3; ++c) {
      *qq++ = drgb[y * nw * 3 + x * 3 + c];
    }
  }
}

encude(drgb2, 256 * 256 * 3, segin);
const double *segout = seg->feed(segin, NULL);
decude(segout, 6, xy);

xy[0] *= 256;
xy[0] += offx;
xy[1] *= 256;
xy[1] += offy;
xy[2] *= 256;
xy[2] += offx;
xy[3] *= 256;
xy[3] += offy;
xy[4] *= 256;
xy[4] += offx;
xy[5] *= 256;
xy[5] += offy;

if (!(xy[0] >= 0 && xy[1] < nw && xy[1] >= 0 && xy[1] < nh && xy[2] >= 0 && xy[2] < nw && xy[3] >= 0 && xy[3] < nh && xy[4] >= 0 && xy[4] < nw && xy[5] >= 0 && xy[5] < nh)) {
xy[0] = nw/2;
xy[1] = nh/2;
xy[2] = nw/2;
xy[3] = nh/2;
xy[4] = nw/2;
xy[5] = nh/2;
}
}

fprintf(stderr, "x=%lf y=%lf\n", xy[0], xy[1]);

SDL_Rect rect;
rect.x = drect.x + s*xy[0] - 5;
rect.y = drect.y + s*xy[1] - 5;
rect.w = 10;
rect.h = 10;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &rect);

rect.x = drect.x + s*xy[2] - 5;
rect.y = drect.y + s*xy[3] - 5;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &rect);

rect.x = drect.x + s*xy[4] - 5;
rect.y = drect.y + s*xy[5] - 5;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &rect);


#if 0
rect.x = w * tgtx - 5;
rect.y = h * tgty - 5;
rect.w = 10;
rect.h = 10;

double speed = 0.03;
tgtx += randgauss() * speed;
tgty += randgauss() * speed;
if (tgtx < 0) tgtx = 0; if (tgtx > 1) tgtx = 1;
if (tgty < 0) tgty = 0; if (tgty > 1) tgty = 1;

isamp = randuint() % nsamp;
memcpy(samprgb + isamp * nw * nh * 3, drgb, sizeof(double) * nw * nh * 3);
memcpy(samptgt + isamp * 2, &tgtx, sizeof(double));
memcpy(samptgt + isamp * 2 + 1, &tgty, sizeof(double));
sampused[isamp] = 1;

unsigned int which = randuint() % nsamp;

if (sampused[which]) {
  encude(samptgt + which * 2, 2, segtgt);
  encude(samprgb + which * nw * nh * 3, nw * nh * 3, segin);
  seg->feed(segin, NULL);
  seg->target(segtgt);
  seg->train(0.001);
fprintf(stderr, "trained\n");
}
#endif

#if 0
SDL_SetRenderDrawColor( renderer, 255, 0, 0, 255 );
SDL_RenderFillRect(renderer, &rect);
#endif

    SDL_RenderPresent( renderer);

SDL_DestroyTexture(text);
SDL_FreeSurface(surf);


    // Pause program so that the window doesn't disappear at once.
    // This willpause for 4000 milliseconds which is the same as 4 seconds
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_KEYUP) {
        int k = event.key.keysym.sym;
        if (k == SDLK_t) tdown = 0;
      }
      if (event.type == SDL_KEYDOWN) {
        int k = event.key.keysym.sym;
        if (k == SDLK_ESCAPE)
          return 0;
        if (k == SDLK_t) tdown = 1;

//        if (k == SDLK_T) {
//          addsample(rgb, tgtx, tgty);
      }
    }
}

}
