#!/usr/bin/perl

sub moretpenc {
  join(' ', map /\s/ || /^</ ? "<" . length : $_, @_) .
  "\n" .
  join('', grep /\s/ || /^</ , @_)
}

print moretpenc(@ARGV) . "\n";
