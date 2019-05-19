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
#include "autoposer.hh"
#include "partrait.hh"
#include "encgendis.hh"
#include "cholo.hh"

using namespace makemore;

static bool *sampused;
static double *samprgb;
static double *samptgt;

struct MFStats {
  double m0, v0, m1, v1;
};

int main( int argc, char* argv[] )
{
  std::vector<MFStats> mfstats, glstats;
  {
  FILE *fp = fopen("mf.tsv", "r");
  std::string line;
  while (readline(fp, &line)) {
    struct MFStats v;
    sscanf(line.c_str(), "%lf\t%lf\t%lf\t%lf\n", &v.m0, &v.v0, &v.m1, &v.v1);
    mfstats.push_back(v);
  }
  fclose(fp);
  }

  {
  FILE *fp = fopen("gl.tsv", "r");
  std::string line;
  while (readline(fp, &line)) {
    struct MFStats v;
    sscanf(line.c_str(), "%lf\t%lf\t%lf\t%lf\n", &v.m0, &v.v0, &v.m1, &v.v1);
    glstats.push_back(v);
  }
  fclose(fp);
  }

double *tmp = new double[1<<20];
 Camera cam;
  cam.open();

  Autoposer autoposer("autoposer.proj");
  assert(autoposer.seginlay->n == 256 * 256 * 3);

  Encgendis egd("big.proj", 1);
  Cholo cholo(egd.ctrlay->n);
  cholo.load(egd.dir + "/cholo.dat");

//  assert(mfstats.size() == egd.ctrlay->n);

seedrand();
double *shift = new double[egd.ctrlay->n];
for (unsigned int j = 0; j < egd.ctrlay->n; ++j)
  shift[j] = randgauss();

    int posX = 0;
    int posY = 0;
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_DisplayMode DM;
    unsigned int w, h;

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
fprintf(stderr, "w=%u h=%u\n", w, h);
 
    // Initialize SDL
    // Create and init the window
    // ==========================================================
    window = SDL_CreateWindow( "Server", posX, posY, w, h, 0 );

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
    SDL_RenderSetLogicalSize( renderer, w, h );
     
    SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );
    SDL_RenderClear( renderer );
    SDL_RenderPresent( renderer);

double tgtx = 0.5, tgty = 0.5;
bool tdown = 0;
bool fdown = 0, mdown = 0, gdown = 0;
int frame = 0;
int mf = 0, gl = 0;

if (argc > 1) frame = atoi(argv[1]);
Partrait campar(cam.w, cam.h);

Pose curpose = Pose::STANDARD;
curpose.scale *= 1.5;
curpose.center.x = (double)campar.w/2.0;
curpose.center.y = (double)campar.h/2.0;
campar.set_pose(curpose);

double asp = (double)w / (double)campar.w;
if (asp > (double)h / (double)campar.h)
  asp = (double)h / (double)campar.h;

SDL_Rect drect;
drect.x = (double)w/2.0 - asp * (double)campar.w/2.0;
drect.y = (double)h/2.0 - asp * (double)campar.h/2.0;
drect.w = asp * (double)campar.w;
drect.h = asp * (double)campar.h;


fprintf(stderr, "w=%u h=%u\ncw=%u ch=%u\n", w, h, cam.w, cam.h);

while (1) {

fprintf(stderr, "tdown=%d\n", tdown);


SDL_SetRenderDrawColor( renderer, 0, 0, 0, 255 );
SDL_RenderClear( renderer );

cam.read(&campar, true);

++frame;
if (tdown) {
  std::string png;
  rgbpng(campar.rgb, campar.w, campar.h, &png);
  char fn[256];
  sprintf(fn, "cam/camera.%08d.png", frame);
  spit(png, fn);
}




//curpose = campar.get_pose();
#if 0
if (curpose.angle > 0.2) curpose.angle = 0.2;
if (curpose.angle < -0.2) curpose.angle = -0.2;
if (curpose.stretch < 0.95) curpose.stretch = 0.95;
if (curpose.stretch > 1.05) curpose.stretch = 1.05;
if (curpose.skew < -0.1) curpose.skew = -0.1;
if (curpose.skew > 0.1) curpose.skew = 0.1;
if (curpose.scale < 64) curpose.scale = 64;
if (curpose.scale > 256) curpose.scale = 256;
#endif

Pose lastpose = campar.get_pose();
curpose.center = lastpose.center;
curpose.angle = lastpose.angle / 2.0;
curpose.skew = lastpose.skew / 2.0;
curpose.stretch = (lastpose.stretch + 1.0) / 2.0;
curpose.scale = (64 + lastpose.scale) / 2.0;

//curpose.angle=0;
//curpose.skew = 0;
//curpose.stretch = 1.0;
//curpose.scale = 64;

campar.set_pose(curpose);
autoposer.autopose(&campar);
Triangle curmark = campar.get_mark();


if (1) {
  Partrait stdpar(256, 256);
  stdpar.set_pose(Pose::STANDARD);
  campar.warp(&stdpar);

  assert(egd.tgtlay->n == stdpar.w * stdpar.h * 3);
  btodv(stdpar.rgb, egd.tgtbuf, egd.tgtlay->n);
  egd.encode();
#if 1

for (unsigned int j = 0; j < egd.ctrlay->n; ++j)
  tmp[j] = egd.ctrbuf[j];

if (gl > 0) {
for (unsigned int k = 0; k < gl; ++k) {
for (unsigned int j = 0; j < egd.ctrlay->n; ++j) {
  tmp[j] = (tmp[j] - glstats[j].m0) / sqrt(glstats[j].v0);
  tmp[j] = tmp[j] * sqrt(glstats[j].v1) + glstats[j].m1;
}
}}

if (mf > 0) {
for (unsigned int k = 0; k < mf; ++k) {
for (unsigned int j = 0; j < egd.ctrlay->n; ++j) {
  tmp[j] = (tmp[j] - mfstats[j].m0) / sqrt(mfstats[j].v0);
  tmp[j] = tmp[j] * sqrt(mfstats[j].v1) + mfstats[j].m1;
}
}}


if (mf < 0) {
for (unsigned int k = 0; k < -mf; ++k) {
for (unsigned int j = 0; j < egd.ctrlay->n; ++j) {
  tmp[j] = (tmp[j] - mfstats[j].m1) / sqrt(mfstats[j].v1);
  tmp[j] = tmp[j] * sqrt(mfstats[j].v0) + mfstats[j].m0;
}
}
}

for (unsigned int j = 0; j < egd.ctrlay->n; ++j)
  egd.ctrbuf[j] = tmp[j];

//cholo.encode(egd.ctrbuf, tmp, true);
//for (unsigned int j = 0; j < egd.ctrlay->n; ++j)
//  tmp[j] = tmp[j] * 1.0;
//cholo.generate(tmp, egd.ctrbuf, true);

  memset(egd.ctxbuf, 0, egd.ctxlay->n * sizeof(double));

egd.ctxbuf[6] = gl;

if (mf < 0) {
  egd.ctxbuf[0] = -mf;
}
if (mf > 0) {
  egd.ctxbuf[1] = mf;
}

//  egd.ctxbuf[6] = 1.0;
#endif
  egd.generate();
  dtobv(egd.tgtbuf, stdpar.rgb, egd.tgtlay->n);

  stdpar.warpover(&campar);
}


SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(campar.rgb, campar.w, campar.h, 24, campar.w * 3, 0xff, 0xff00, 0xff0000, 0);
SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, surf);

SDL_Rect srect;
srect.x = 0;
srect.y = 0;
srect.w = campar.w;
srect.h = campar.h;
SDL_RenderCopy(renderer, text, &srect, &drect);

#if 0
srect.x = drect.x + asp*curmark.p.x - 5;
srect.y = drect.y + asp*curmark.p.y - 5;
srect.w = 10;
srect.h = 10;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &srect);

srect.x = drect.x + asp*curmark.q.x - 5;
srect.y = drect.y + asp*curmark.q.y - 5;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &srect);

srect.x = drect.x + asp*curmark.r.x - 5;
srect.y = drect.y + asp*curmark.r.y - 5;
SDL_SetRenderDrawColor( renderer, 0, 0, 255, 255 );
SDL_RenderFillRect(renderer, &srect);
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
        if (k == SDLK_f) fdown = 0;
        if (k == SDLK_m) mdown = 0;
        if (k == SDLK_g) gdown = 0;
      }
      if (event.type == SDL_KEYDOWN) {
        int k = event.key.keysym.sym;
        if (k == SDLK_ESCAPE)
          return 0;
        if (k == SDLK_t) tdown = 1;
        if (k == SDLK_f) { if (!fdown) { ++mf; } fdown = 1; }
        if (k == SDLK_m) { if (!mdown) { --mf; } mdown = 1; }
        if (k == SDLK_g) { if (!gdown) { gl = !gl; } gdown = 1; }

//        if (k == SDLK_T) {
//          addsample(rgb, tgtx, tgty);
      }
    }
}

}
