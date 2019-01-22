#!/usr/bin/perl

local $/ = \232;
while (<>) {
  my $lab = substr($_, 40, 192);
  print join("\t", stats($lab)) . "\n";
}

sub stats {
  my $lab = shift;
  my @lab = unpack('C*', $lab);

  my @edgesum;
  my $edgecount = 0;
  for my $y (0 .. 7) {
    for my $x (0 .. 7) {
      next unless $x == 0 || $y == 0 || $x == 7;
      for my $c (0 .. 2) {
        $edgesum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$edgecount;
    }
  }

  my @edgemean = map {$_/$edgecount} @edgesum;
  my @edgevar;

  for my $y (0 .. 7) {
    for my $x (0 .. 7) {
      next unless $x == 0 || $y == 0 || $x == 7;
      for my $c (0 .. 2) {
        my $q = $lab[$y * 24 + $x * 3 + $c] - $edgemean[$c];
        $edgevar[$c] += $q * $q;
      }
    }
  }

  $_ /= $edgecount for @edgevar;
  $_ = sqrt($_) for @edgevar;
  $_ *= 2 for @edgevar;


  my @centersum;
  my $centercount = 0;
  for my $y (0 .. 7) {
    next unless $y > 0 && $y < 7;
    for my $x (0 .. 7) {
      next unless $x > 1 && $x < 6;
      for my $c (0 .. 2) {
        $centersum[$c] += $lab[$y * 8 * 3 + $x * 3 + $c];
      }
      ++$centercount;
    }
  }

  my @centermean = map {$_/$centercount} @centersum;
  my @centervar;

  for my $y (0 .. 7) {
    next unless $y > 0 && $y < 7;
    for my $x (0 .. 7) {
      next unless $x > 1 && $x < 6;
      for my $c (0 .. 2) {
        my $q = $lab[$y * 24 + $x * 3 + $c] - $centermean[$c];
        $centervar[$c] += $q * $q;
      }
    }
  }

  $_ /= $centercount for @centervar;
  $_ = sqrt($_) for @centervar;
  $_ *= 2 for @centervar;

  my @stats = (@edgemean, @edgevar, @centermean, @centervar);
  @stats
}
