#define __MAKEMORE_IO_CC__ 1
#include "io.hh"
#include "process.hh"
#include "session.hh"

namespace makemore {

bool IO::put(const strvec &vec) {
//fprintf(stderr, "got put [%s] n=%u m=%u\n", joinwords(vec).c_str(), n, m);
  if (n >= m)
    return false;

  if (n == 0) {
    wake_readers();
  }
  ++n;
  q.push_back(vec);

  return true;
}

strvec *IO::get() {
//fprintf(stderr, "got get [%s] n=%u m=%u\n", "", n, m);
  if (n == 0) {
    return NULL;
  }

  if (n == l) {
    wake_writers();
  }

  auto i = q.begin();
  assert(i != q.end());
  x = *i;
  q.erase(i);
  --n;
  assert(n < m);

  return &x;
}

strvec *IO::peek() {
  if (n == 0)
    return NULL;

  auto i = q.begin();
  assert(i != q.end());
  strvec &y = *i;

  return &y;
}

void IO::wake_readers() {
  for (Process *reader = head_process_reader; reader; reader = reader->next_reader)
    if (reader->mode == Process::MODE_READING)
      reader->wake();
}

void IO::wake_writers() {
  for (Process *writer = head_process_writer; writer; writer = writer->next_writer)
    if (writer->mode == Process::MODE_WRITING)
      writer->wake();
}

void IO::link_reader(Session *reader) {
  assert(reader);

  reader->next_reader = head_session_reader;
  reader->prev_reader = NULL;
  if (head_session_reader)
    head_session_reader->prev_reader = reader;
  head_session_reader = reader;

  ++nreaders;
}

void IO::link_reader(Process *reader) {
  assert(reader);

  reader->next_reader = head_process_reader;
  reader->prev_reader = NULL;
  if (head_process_reader)
    head_process_reader->prev_reader = reader;
  head_process_reader = reader;

  ++nreaders;
}


void IO::unlink_reader(Session *reader) {
  assert(reader);

  if (reader->next_reader)
    reader->next_reader->prev_reader = reader->prev_reader;
  if (reader->prev_reader)
    reader->prev_reader->next_reader = reader->next_reader;
  if (reader == head_session_reader)
    head_session_reader = reader->next_reader;

  assert(nreaders);
  --nreaders;
  if (nreaders == 0) {
    wake_writers();
  }

  autodestruct();
}

void IO::unlink_reader(Process *reader) {
  assert(reader);

  if (reader->next_reader)
    reader->next_reader->prev_reader = reader->prev_reader;
  if (reader->prev_reader)
    reader->prev_reader->next_reader = reader->next_reader;
  if (reader == head_process_reader)
    head_process_reader = reader->next_reader;
    wake_writers();

  assert(nreaders);
  --nreaders;
  if (nreaders == 0) {
    wake_writers();
  }

  autodestruct();
}


void IO::link_writer(Session *writer) {
  assert(writer);

  writer->next_writer = head_session_writer;
  writer->prev_writer = NULL;
  if (head_session_writer)
    head_session_writer->prev_writer = writer;
  head_session_writer = writer;

  ++nwriters;
}

void IO::link_writer(Process *writer) {
  assert(writer);

  writer->next_writer = head_process_writer;
  writer->prev_writer = NULL;
  if (head_process_writer)
    head_process_writer->prev_writer = writer;
  head_process_writer = writer;

  ++nwriters;
}


void IO::unlink_writer(Session *writer) {
  assert(writer);

  if (writer->next_writer)
    writer->next_writer->prev_writer = writer->prev_writer;
  if (writer->prev_writer)
    writer->prev_writer->next_writer = writer->next_writer;
  if (writer == head_session_writer)
    head_session_writer = writer->next_writer;

  assert(nwriters);
  --nwriters;

  if (nwriters == 0 && !can_get()) {
    wake_readers();
  }

  autodestruct();
}

void IO::unlink_writer(Process *writer) {
  assert(writer);

  if (writer->next_writer)
    writer->next_writer->prev_writer = writer->prev_writer;
  if (writer->prev_writer)
    writer->prev_writer->next_writer = writer->next_writer;
  if (writer == head_process_writer)
    head_process_writer = writer->next_writer;

  assert(nwriters);
  --nwriters;

  if (nwriters == 0 && !can_get()) {
    wake_readers();
  }
  autodestruct();
}

}
