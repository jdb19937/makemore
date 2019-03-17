#define __MAKEMORE_IO_CC__ 1
#include "io.hh"
#include "process.hh"
#include "session.hh"
#include "word.hh"

namespace makemore {

using namespace std;

bool IO::put(const strvec &sv) {
  Line *wp = new Line;
  strvec_to_line(sv, wp);
  return this->put(wp);
}

bool IO::put(Line *vec) {
  if (n >= m)
    return false;

  if (n == 0) {
    wake_readers();
  }
  ++n;
  q.push_back(vec);

  return true;
}

Line *IO::get() {
//fprintf(stderr, "got get [%s] n=%u m=%u\n", "", n, m);
  if (n == 0) {
    return NULL;
  }

  if (n == l) {
    wake_writers();
  }

  auto i = q.begin();
  assert(i != q.end());
  Line *x = *i;
  q.erase(i);
  assert(n);
  --n;

  return x;
}

Line *IO::peek() {
  if (n == 0)
    return NULL;

  auto i = q.begin();
  assert(i != q.end());
  vector<Word> *y = *i;

  return y;
}

void IO::wake_readers() {
  for (auto reader : process_readers) {
    if (reader->mode == Process::MODE_READING) {
      assert(reader->waitfd >= 0);
      assert(reader->waitfd < reader->itab.size());

      if (reader->itab[reader->waitfd] == this) {
        reader->wake();
      }
    }
  }
}

void IO::wake_writers() {
  for (auto writer : process_writers) {
    if (writer->mode == Process::MODE_WRITING) {
      assert(writer->waitfd >= 0);
      assert(writer->waitfd < writer->otab.size());

      if (writer->otab[writer->waitfd] == this) {
        writer->wake();
      }
    }
  }
}

void IO::link_reader(Session *reader) {
  session_readers.push_back(reader);
  ++nreaders;
}

void IO::link_reader(Process *reader) {
  process_readers.push_back(reader);
  ++nreaders;
}


void IO::unlink_reader(Session *reader) {
  assert(1 == listerase(session_readers, reader));
  assert(nreaders);
  --nreaders;

  if (nreaders == 0) {
    wake_writers();
  }

  autodestruct();
}

void IO::unlink_reader(Process *reader) {
  assert(1 == listerase(process_readers, reader));

  assert(nreaders);
  --nreaders;

  if (nreaders == 0) {
    wake_writers();
  }

  autodestruct();
}


void IO::link_writer(Session *writer) {
  session_writers.push_back(writer);
  ++nwriters;
}

void IO::link_writer(Process *writer) {
  process_writers.push_back(writer);
  ++nwriters;
}


void IO::unlink_writer(Session *writer) {
  assert(1 == listerase(session_writers, writer));

  assert(nwriters);
  --nwriters;

  if (nwriters == 0 && !can_get()) {
    wake_readers();
  }

  autodestruct();
}

void IO::unlink_writer(Process *writer) {
  assert(1 == listerase(process_writers, writer));

  assert(nwriters);
  --nwriters;

  if (nwriters == 0 && !can_get()) {
    wake_readers();
  }
  autodestruct();
}

}
