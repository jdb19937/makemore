#ifndef __MAKEMORE_DISPLAY_HH__
#define __MAKEMORE_DISPLAY_HH__ 1

#include <SDL2/SDL.h>

namespace makemore {

struct Display {
  class SDL_Window *window;
  class SDL_Renderer *renderer;
  void *dm;

  unsigned int w, h;

  Display();
  ~Display();

  void open();
  void close();

  void fill_black();
  void draw_pigeon(unsigned int x, unsigned int y);
  void update(const class Partrait &);

  void present();
};

}

#endif
