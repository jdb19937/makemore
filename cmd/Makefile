CXX = g++
CXXFLAGS = -g

.PHONY: all
all: cmd_echo.so cmd_sh.so cmd_cut.so cmd_then.so cmd_yes.so cmd_head.so cmd_grep.so cmd_pre.so cmd_post.so cmd_cols.so cmd_sort.so cmd_uniq.so cmd_pick.so cmd_pass.so cmd_exit.so cmd_be.so cmd_make.so cmd_nomgen.so cmd_setword.so cmd_pack.so cmd_unpack.so cmd_setline.so cmd_whoami.so cmd_setgrid.so cmd_getgrid.so cmd_first.so cmd_catrows.so cmd_catcols.so cmd_encude.so cmd_decude.so cmd_culist.so cmd_cufree.so cmd_savegrid.so cmd_saveline.so cmd_saveword.so cmd_count.so cmd_wcl.so

%.so: %.cc
	$(CXX) -o $@ $(CXXFLAGS) -shared -I.. -fPIC $<

.PHONY: clean
clean:
	rm -f *.so
