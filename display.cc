#define __MAKEMORE_DISPLAY_CC__ 1
#include "display.hh"

#include <stdio.h>
#include <assert.h>

#include <SDL2/SDL.h>

#include "partrait.hh"

namespace makemore {

Display::Display() {
  window = NULL;
  renderer = NULL;
  dm = NULL;
  w = 0;
  h = 0;
}

Display::~Display() {
  close();
}

void Display::open() {
  int err;
  err = SDL_Init(SDL_INIT_EVERYTHING);
  assert(err == 0);

  assert(!dm);
  dm = new SDL_DisplayMode();
  err = SDL_GetCurrentDisplayMode(0, (SDL_DisplayMode *)dm);
  assert(err == 0);
  w = ((SDL_DisplayMode *)dm)->w;
  h = ((SDL_DisplayMode *)dm)->h;
fprintf(stderr, "w=%u h=%u\n", w, h);

  assert(!window);
  window = SDL_CreateWindow("makemore", 0, 0, w, h, 0);
  assert(window);

  err = SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
  assert(err == 0);

  assert(!renderer);
  renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  assert(renderer);

  SDL_RenderSetLogicalSize(renderer, w, h );
}

void Display::close() {
  int err;

  assert(renderer);
  SDL_DestroyRenderer(renderer);
  renderer = NULL;

  assert(window);
  SDL_DestroyWindow(window);
  assert(err == 0);
  window = NULL;

  assert(dm);
  delete ((SDL_DisplayMode *)dm);
  dm = NULL;
}

void Display::update(const Partrait &prt) {
  update(prt.rgb, prt.w, prt.h);
}

void Display::update(const uint8_t *rgb, unsigned int pw, unsigned int ph) {
  double asp = (double)w / (double)pw;
  if (asp > (double)h / (double)ph)
    asp = (double)h / (double)ph;

  SDL_Rect drect;
  drect.x = (double)w / 2.0 - asp * (double)pw / 2.0;
  drect.y = (double)h / 2.0 - asp * (double)ph / 2.0;
  drect.w = asp * (double)pw;
  drect.h = asp * (double)ph;
  SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(
    (void *)rgb, pw, ph, 24, pw * 3, 0xff, 0xff00, 0xff0000, 0
  );

  SDL_Texture *text = SDL_CreateTextureFromSurface(renderer, surf);

  SDL_Rect srect;
  srect.x = 0;
  srect.y = 0;
  srect.w = pw;
  srect.h = ph;
  SDL_RenderCopy(renderer, text, &srect, &drect);

  SDL_DestroyTexture(text);
  SDL_FreeSurface(surf);
}

void Display::fill_black() {
  SDL_Rect srect;

  srect.w = w;
  srect.h = h;
  srect.x = 0;
  srect.y = 0;

  SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
  SDL_RenderFillRect(renderer, &srect);
}

void Display::draw_pigeon(unsigned int x, unsigned int y) {
  SDL_Rect srect;

  srect.w = 10;
  srect.h = 10;
  srect.x = (int)x - srect.w / 2;
  srect.y = (int)y - srect.h / 2;

  SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255 );
  SDL_RenderFillRect(renderer, &srect);
}

void Display::present() {
  SDL_RenderPresent(renderer);
}

}
