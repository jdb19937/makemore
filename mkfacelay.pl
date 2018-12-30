#!/usr/bin/perl -s

use strict;

die unless @ARGV == 2;
my ($dim, $chan) = @ARGV;

select(STDOUT);
$| = 1;

my (@ax, @ay, @ar);

for (my $y = 0; $y < $dim; ++$y) {
  for (my $x = 0; $x < $dim; ++$x) {

    my $ax = ($x + 0.5) / $dim;
    my $ay = ($y + 0.5) / $dim;
    my $ar = 0.5 / $dim;

    for (1 .. $chan) {
      push @ax, $ax;
      push @ay, $ay;
      push @ar, $ar;
    }
  }
}

print pack('N', $dim * $dim * $chan);
print pack('d*', @ax);
print pack('d*', @ay);
print pack('d*', @ar);
