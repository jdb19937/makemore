#!/usr/bin/perl -s

use FindBin qw($Bin);
my $program = join(' ', @ARGV);

use File::Temp qw(tempdir);

my $pipe = "";
if ($::dim == 8) {
  $pipe .= "| ./labshrink 128 128 4";
} elsif ($::dim == 16) {
  $pipe .= "| ./labshrink 128 128 3";
} elsif ($::dim == 32) {
  $pipe .= "| ./labshrink 128 128 2";
} elsif ($::dim == 64) {
  $pipe .= "| ./labshrink 128 128 1";
} elsif ($::dim == 128) {

} else {
  die "need dim parameter (8, 16, 32, 64, 128)";
}

my ($chan, $laydim);
if ($::gray)  {
  $pipe .= "| ./labtogray $dim $dim";
  if ($::hifreq) {
    $chan = 3;
    $laydim = $dim / 2;
    $pipe .= "| ./grayhifreq $dim $dim";
  } else {
    $chan = 1;
    $laydim = $dim;
  }
} else {
  if ($::hifreq) {
    $chan = 9;
    $laydim = $dim / 2;
    $pipe .= "| ./labhifreq $dim $dim";
  } else {
    $chan = 3;
    $laydim = $dim;
  }
}

select(STDOUT);
$| = 1;

my (@ax, @ay, @ar);

for (my $y = 0; $y < $laydim; ++$y) {
  for (my $x = 0; $x < $laydim; ++$x) {

    my $ax = ($x + 0.5) / $laydim;
    my $ay = ($y + 0.5) / $laydim;
    my $ar = sqrt(1.0 / atan2(0,-1)) / $laydim;

    for (1 .. $chan) {
      push @ax, $ax;
      push @ay, $ay;
      push @ar, $ar;
    }
  }
}

print pack('N', $laydim * $laydim * $chan);
print pack('d*', @ax);
print pack('d*', @ay);
print pack('d*', @ar);
print pack('N', 202599);

my $celeba = "$Bin/celeba-dataset";
my $attrfn = "$celeba/list_attr_celeba.csv";
open(my $attrfp, "< $attrfn") or die "$0: $attrfn: $!";
scalar <$attrfp>;

my $count = 0;

for my $id ('000001' .. '202599') {
  my $jpgfn = "$celeba/img_align_celeba/$id.jpg";
  my $jpgdata;
  {
    open(my $jpgfp, "< $jpgfn") or die "$0: $jpgfn: $!";
    undef local $/;
    $jpgdata = <$jpgfp>;
  }

  open(my $outfp, "| djpeg -ppm |pnmcut -top 20 -bottom 197 |pnmscale -width 128 |./ppmtolab $pipe") or die;
  print $outfp $jpgdata;
  close($outfp);

  ++$count;
  if (defined($::max) && $count >= $::max) {
    print STDERR "all done count=$count\n";
    last;
  }
}
