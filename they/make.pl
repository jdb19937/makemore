my $c = shift(@ARGV);
my $d = ord(lc chr($c));
srand($d);

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
  if ($y >= 6) { $y = 6; }
  if ($y < 3) { $y = 3; }

  my $nc = $y * 5 + $x;
  $map[$nc] = 1;
}

for (1 .. 3) {
  my $c = geton();
  $map[$c] = 0;
}

if ($c == ord('p') || $c == ord('P')) {
  $map[4 * 5 + 0] = 1;
  $map[5 * 5 + 0] = 1;
  $map[6 * 5 + 0] = 1;
}
if ($c == ord('i') || $c == ord('I')) {
  $map[4 * 5 + 0] = 1;
  $map[5 * 5 + 0] = 1;
  $map[5 * 5 + 1] = 1;
  $map[5 * 5 + 2] = 1;
}
if ($c == ord('r') || $c == ord('R')) {
  $map[5 * 5 + 0] = 1;
  $map[5 * 5 + 1] = 1;
  $map[5 * 5 + 2] = 1;
}
if ($c == ord('l') || $c == ord('L')) {
  $map[3 * 5 + 4] = 1;
  $map[3 * 5 + 3] = 1;
  $map[4 * 5 + 3] = 1;
  $map[5 * 5 + 3] = 1;
  $map[6 * 5 + 3] = 1;
}

if ($c == ord('o') || $c == ord('O')) {
  $map[4 * 5 + 0] = 1;
  $map[4 * 5 + 1] = 1;
  $map[4 * 5 + 2] = 1;
}
if ($c == ord('y') || $c == ord('Y')) {
  $map[4 * 5 + 0] = 1;
  $map[4 * 5 + 1] = 1;
  $map[4 * 5 + 2] = 1;
}
if ($c == ord('a') || $c == ord('A')) {
  $map[4 * 5 + 0] = 1;
  $map[4 * 5 + 1] = 1;
  $map[5 * 5 + 1] = 1;
  $map[6 * 5 + 1] = 1;
  $map[4 * 5 + 2] = 1;
}

if ($c >= ord('A') && $c <= ord('Z') || $c >= ord('0') && $c <= ord('9')) {
  $map[$_] = 1 for 5 .. 9;
  if ($c >= ord('A') && $c <= ord('Z')) {
    $map[11] = 1;
    $map[13] = 1;
  }
}

if ($c == ord(' ') || $c == ord("\t")) {
  @map = (0) x 35;
}

print chr(0) x 48;
for my $y (0 .. 20) {
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
print chr(0) x 48;
print chr(0) x 48;
