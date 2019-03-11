CXX = g++
CXXFLAGS = -g

.PHONY: all
all: cmd_echo.so cmd_sh.so cmd_cut.so cmd_then.so cmd_yes.so cmd_head.so cmd_grep.so cmd_prepend.so cmd_append.so cmd_cols.so cmd_sort.so cmd_uniq.so

%.so: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -shared -I.. -fPIC $<

.PHONY: clean
clean:
	rm -f *.so
