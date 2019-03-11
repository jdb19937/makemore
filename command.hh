#ifndef __MAKEMORE_COMMAND_HH__
#define __MAKEMORE_COMMAND_HH__ 1

#include <string>
#include <vector>

namespace makemore {

typedef void (*Command)( class Process *);

extern Command find_command(const std::string &cmd);

}

#endif
