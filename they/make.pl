my $c = shift(@ARGV);
srand($c);

print "P6\n16 24\n255\n";

my @map = (0) x 35;
$map[4 * 5 + 2] = 1;

sub geton {
  my @x;
  for my $i (0 .. 34) {
    if ($map[$i]) {
      push @x, $i;
    }
  }
  @x = sort { rand(1) <=> 0.5 } @x;
  $x[0]
}

for (1 .. 20) {
  my $c = geton();

  my $x = $c % 5;
  my $y = int($c / 5);

  my $q = int rand(4);
  if ($q == 0) {
    ++$y;
  } elsif ($q == 1) {
    --$y;
  } elsif ($q == 2) {
    ++$x;
  } else {
    --$x;
  }
  if ($x >= 4) { $x = 4; }
  if ($x < 0) { $x = 0; }
  if ($y >= 7) { $y = 7; }
  if ($y < 3) { $y = 3; }

  my $nc = $y * 5 + $x;
  $map[$nc] = 1;
}

for (1 .. 3) {
  my $c = geton();
  $map[$c] = 0;
}

if ($c >= ord('A') && $c <= ord('Z') || $c >= ord('0') && $c <= ord('9')) {
  $map[$_] = 1 for 0 .. 4;
  $map[6] = 1;
  $map[8] = 1;
}

for my $y (0 .. 23) {
  for my $x (0 .. 14) {
    my $mx = int($x / 3);
    my $my = int($y / 3);

    my $v = 0;
    if ($map[$my * 5 + $mx]) {
      $v = 0x55;
    }
    print chr($v) x 3;
  }
  print chr(0) x 3;
}
