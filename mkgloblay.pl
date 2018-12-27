#!/usr/bin/perl

my ($n) = @ARGV;

for (1 .. $n) {
  my $ax = 0.5;
  my $ay = 0.5;
  my $ar = sqrt(2) / 2;
  print(pack('d3', $ax, $ay, $ar));
}
