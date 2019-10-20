#!/usr/bin/perl

system("ppmmake black 16 8 > tmp/chop.ppm");
system("ppmmake black 16 1 > tmp/row.ppm");

for (my $c = 0; $c < 256; ++$c) {
  my $x = int($c % 16) * 16;
  my $y = int($c / 16) * 16;

  system("perl ./make.pl $c > bynr/$c.ppm");
}

for my $y (0 .. 15) {
  my @row;
  for my $x (0 .. 15) {
    my $c = $y * 16 + $x;
    push @row, "bynr/$c.ppm";
  }
  system("pnmcat -lr " . join(" ", @row) . " > rows/row$y.ppm") and die;
}

my $rp = join(' ', map "rows/row${_}.ppm", 0 .. 15);
system("pnmcat -tb $rp > tmp/they_8.ppm");

system("cat tmp/they_8.ppm | convert ppm:- -define png:color-type=2 png:- > they_8.png");
system("cat tmp/they_8.ppm | pnmscale 1.5 | convert ppm:- -define png:color-type=2 png:- > they_12.png");
system("cat tmp/they_8.ppm | pnmscale 2 | convert ppm:- -define png:color-type=2 png:- > they_16.png");
