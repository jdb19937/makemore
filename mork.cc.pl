#!/usr/bin/perl

my @black = (0,0,0);
my @red = (0xFF, 0, 0);
my @pink = (0xFF, 0x7F, 0x7F);
my @green = (0, 0xFF, 0);
my @gray = (0x7F, 0x7F, 0x7F);
my @white = (0xff, 0xff, 0xff);
my @lightblue = (0x7f,0x7f,0xff);
my @darkblue = (0,0,0x7f);
my @blue = (0,0,0xff);
my @green = (0,0xff,0);
my @darkgreen = (0,0x7f,0);
my @yellow = (0xff,0xff,0);
my @brown = (0x7f,0x7f,0);
my @darkred = (0x7f,0,0);
my @mag = (0xff, 0, 0x7f);
my @purple = (0x7f, 0, 0x7f);
my @orange = (0xff, 0x7f, 0);
my @cyan = (0,0xff,0xff);

$x{'0'} = [
  @green, @black,
  @black, @black,
  @white, @white
];
$x{'1'} = [
  @green, @black,
  @black, @white,
  @black, @white,
];
$x{'2'} = [
  @green, @black,
  @black, @white,
  @white, @black,
];
$x{'3'} = [
  @green, @black,
  @white, @black,
  @black, @white,
];
$x{'4'} = [
  @green, @black,
  @white, @white,
  @black, @black,
];
$x{'5'} = [
  @green, @black,
  @white, @black,
  @white, @black,
];
$x{'6'} = [
  @green, @white,
  @black, @black,
  @black, @white,
];
$x{'7'} = [
  @green, @white,
  @black, @black,
  @white, @black,
];
$x{'8'} = [
  @green, @white,
  @black, @white,
  @black, @black,
];
$x{'9'} = [
  @green, @white,
  @white, @black,
  @black, @black,
];

$x{'a'} = [
  @black, @black,
  @red,   @red,
  @darkred,   @darkred,
];

$x{'b'} = [
  @black, @black,
  @blue, @blue,
  @blue, @blue,
];

$x{'c'} = [
  @black, @black,
  @yellow, @brown,
  @yellow, @brown
];

$x{'d'} = [
  @black, @black,
  @darkgreen, @blue,
  @blue, @darkgreen,
];

$x{'e'} = [
  @black, @black,
  @green, @green,
  @darkgreen, @darkgreen,
];

$x{'f'} = [
  @black, @black,
  @yellow, @yellow,
  @yellow, @brown,
];

$x{'g'} = [
  @black, @black,
  @green, @darkgreen,
  @green, @darkgreen,
];

$x{'i'} = [
  @black, @black,
  @yellow, @yellow,
  @brown, @brown
];

$x{'h'} = [
  @black, @black,
  @white, @gray,
  @white, @gray,
];
$x{'j'} = [
  @black, @black,
  @yellow, @darkgreen,
  @darkgreen, @darkgreen,
];
$x{'k'} = [
  @black, @black,
  @blue, @darkblue,
  @blue, @darkblue,
];

$x{'l'} = [
  @black, @black,
  @purple, @brown,
  @purple, @purple
];
  

$x{'m'} = [
  @black, @black,
  @mag, @mag,
  @mag, @mag,
];

$x{'n'} = [
  @black, @black,
  @orange, @orange,
  @brown, @orange,
];

$x{'o'} = [
  @black, @black,
  @white, @white,
  @gray, @gray,
];

$x{'p'} = [
  @black, @black,
  @yellow, @yellow,
  @yellow, @yellow
];

$x{'q'} = [
  @black, @black, 
  @purple, @blue,
  @blue, @purple
];

$x{'r'} = [
  @black, @black,
  @red, @red,
  @red, @darkred
];

$x{'s'} = [
  @black, @black,
  @orange, @red,
  @red, @orange,
];

$x{'t'} = [
  @black, @black,
  @orange, @yellow,
  @yellow, @orange,
];

$x{'u'} = [
  @black, @black,
  @blue, @blue,
  @darkblue,@darkblue
];


$x{'v'} = [
  @black, @black,
  @darkgreen, @green,
  @green, @green,
];
$x{'w'} = [
  @black, @black,
  @cyan, @cyan,
  @cyan, @cyan,
];

$x{'x'} = [
  @black, @black,
  @red, @green,
  @green, @red,
];

$x{'y'} = [
  @black, @black,
  @yellow, @yellow,
  @darkgreen, @darkgreen,
];
$x{'z'} = [
  @black, @black,
  @purple, @yellow,
  @yellow, @purple,
];

$x{'.'} = [
  @black, @black,
  @black, @black,
  @white, @black,
];
$x{','} = [
  @black, @black,
  @black, @black,
  @gray, @black,
];
$x{';'} = [
  @black, @black,
  @white, @black,
  @gray, @black,
];
$x{':'} = [
  @black, @black,
  @white, @black,
  @white, @black,
];
$x{'"'} = [
  @white, @white,
  @black, @black,
  @black, @black,
];
$x{"'"} = [
  @gray, @gray,
  @black, @black,
  @black, @black,
];
$x{"/"} = [
  @black, @white,
  @white, @white,
  @white, @black,
];
$x{"\\"} = [
  @white, @black,
  @white, @white,
  @black, @white,
];
$x{"<"} = [
  @black, @white,
  @white, @black,
  @black, @white,
];
$x{">"} = [
  @white, @black,
  @black, @white,
  @white, @black,
];

$x{"\0"} = [(32) x 18];

$x{'{'} = [
  @gray, @blue,
  @gray, @black,
  @gray, @black,
];
$x{'['} = [
  @gray, @green,
  @gray, @black,
  @gray, @black,
];
$x{'('} = [
  @white, @yellow,
  @white, @black,
  @white, @black,
];

$x{'`'} = [
  @white, @black,
  @black, @black,
  @black, @black,
];

$x{'|'} = [
  @yellow, @black,
  @yellow, @black,
  @yellow, @black,
];

$x{'}'} = [
  @blue, @gray,
  @black, @gray,
  @black, @gray,
];
$x{']'} = [
  @green, @gray,
  @black, @gray,
  @black, @gray,
];
$x{')'} = [
  @yellow, @white,
  @black, @white,
  @black, @white,
];

$x{'^'} = [
  @purple, @purple,
  @black, @black,
  @black, @black,
];

$x{'~'} = [
  @black, @black,
  @yellow, @yellow,
  @black, @black,
];

$x{'='} = [
  @black, @black,
  @green, @green,
  @black, @black,
];
$x{'+'} = [
  @black, @black,
  @yellow, @blue,
  @black, @black,
];
$x{'!'} = [
  @red, @black,
  @red, @black,
  @white, @black,
];
$x{'?'} = [
  @mag, @mag,
  @black, @mag,
  @white, @black,
];
$x{'@'} = [
  @black, @black,
  @brown, @brown,
  @black, @black,
];
$x{'#'} = [
  @black, @black,
  @orange, @orange,
  @black, @black,
];
$x{'*'} = [
  @black, @black,
  @purple, @green,
  @black, @black,
];
$x{'$'} = [
  @black, @black,
  @darkgreen, @darkgreen,
  @black, @black,
];
$x{'%'} = [
  @black, @black,
  @darkred, @darkred,
  @black, @black,
];
$x{'&'} = [
  @blue, @black,
  @blue, @black,
  @blue, @black,
];

$x{'_'} = [
  @black, @black,
  @black, @black,
  @white, @white,
];
$x{'-'} = [
  @black, @black,
  @white, @white,
  @black, @black,
];

$x{' '} = [(@black) x 6];


for (grep defined $x{$_} && !defined $x{uc $_}, 'a' .. 'z') {
  my @q = @{$x{$_}};
  for my $i (0..5) { $q[$i] = $q[$i + 6]; }
  $x{uc $_} = \@q;
}

my @k = sort keys %x;

my @map;
for (0 .. 255) {
  if ($x{chr($_)}) {
    $map[$_] = $x{chr($_)};
  } else {
    $map[$_] = [ map { int rand(128) } 0 .. 17 ];
  }
};

#@k = split //, '0123456789 Dan Brumleve abcdefghijklmnopqrstuvwxyz <>,./;:"[]{}()|?_-+=!@#$%^&*' . "'\\`~";
my $w = 2 * 16;
my $h = 3 * 16;

print "#define __MAKEMORE_MORK_CC__ 1\n";
print "#include \"mork.hh\"\n";
print "\n";
print "namespace makemore {\n";
print "\n";
print "const uint8_t mork_rgb[32 * 48 * 3] = {\n";

my @out = (0) x ($w * $h * 3);
@out == 32 * 48 * 3 or die;

for (my $i = 0; $i < 256; ++$i) {
  my @z = @{$map[$i]};
  my $x0 = 2 * int($i % 16);
  my $y0 = 3 * int($i / 16);
  my $j = 0;
  for (my $y = $y0; $y < $y0 + 3; ++$y) {
    for (my $x = $x0; $x < $x0 + 2; ++$x) {
      for my $c (0 .. 2) {
        $out[3 * $y * $w + 3 * $x + $c] = $z[$j++];
      }
    }
  }
}

print join(",", @out);
print "};\n";
print "\n";
print "}\n";
