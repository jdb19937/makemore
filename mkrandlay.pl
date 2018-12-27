#!/usr/bin/perl

my ($n, $mul) = @ARGV;
$mul //= 1.0;

for (1 .. $n) {
  my $ax = rand(1);
  my $ay = rand(1);
  my $ar = sqrt($mul / (atan2(0,-1) * $n));
  print(pack('d3', $ax, $ay, $ar));
}
