#include <system.hh>
#include <urbite.hh>
#include <process.hh>
#include <strutils.hh>

using namespace makemore;

extern "C" void mainmore(Process *);

static FILE *fp = NULL;

void mainmore(
  Process *process
) {
  if (!fp) { fp = fopen("vid.out", "w"); }
  fwrite(process->args[0].data(), 1, process->args[0].length(), fp);
//fprintf(stderr, "here echo args=%s\n", joinwords(process->args).c_str());

  strvec out;
  out.resize(1);
  out[0] = "ok";
  (void)process->write(out);
}
