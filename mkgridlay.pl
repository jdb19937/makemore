#!/usr/bin/perl

my ($dim, $mul) = @ARGV;
$mul //= 1.0;

for (my $y = 0; $y < $dim; ++$y) {
  for (my $x = 0; $x < $dim; ++$x) {

    my $ax = ($x + 0.5) / $dim;
    my $ay = ($y + 0.5) / $dim;
    my $ar = sqrt($mul / atan2(0,-1)) / $dim;

    print(pack('d3', $ax, $ay, $ar));
  }
}
