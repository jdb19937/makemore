CXX = g++

.PHONY: all
all: cmd_echo.so cmd_sh.so cmd_then.so cmd_cut.so

%.so: %.cc
	$(CXX) -o $@ -shared -I.. -fPIC $<

.PHONY: clean
clean:
	rm -f *.so
