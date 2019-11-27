#define __MAKEMORE_WALL_CC__ 1
#include "wall.hh"
#include "strutils.hh"

namespace makemore {

void Wall::load(const std::string &fn) {
  clear();
  std::string txt = makemore::slurp(fn);
  split(txt, '\n', &posts);
}
void Wall::load(FILE *fp) {
  clear();
  std::string txt = makemore::slurp(fp);
  split(txt, '\n', &posts);
}

void Wall::save(const std::string &fn) {
  std::string str;
  if (posts.size() > 0) {
    str = join(posts, '\n') + "\n";
  } else {
    str = "";
  }
  makemore::spit(str, fn);
}

bool Wall::erase(unsigned int i, const std::string &hash) {
  if (i >= posts.size())
    return false;
  if (sha256(posts[i]) != hash)
    return false;
  
  unsigned int n = posts.size();
  for (unsigned int j = i + 1; j < n; ++j)
    posts[j - 1] = posts[j];
  posts.resize(n - 1);

  return true;
}

bool Wall::erase(unsigned int i) {
  if (i >= posts.size())
    return false;
  
  unsigned int n = posts.size();
  for (unsigned int j = i + 1; j < n; ++j)
    posts[j - 1] = posts[j];
  posts.resize(n - 1);

  return true;
}

void Wall::truncate(unsigned int m) {
  if (posts.size() <= m)
    return;

  unsigned int n = posts.size();
  unsigned int d = n - m;

  std::vector<std::string> posts_bak = posts;
  posts.resize(m);
  for (unsigned int i = 0; i < m; ++i)  {
    posts[i] = posts_bak[i + d];
  }
};

}
