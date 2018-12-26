#!/usr/bin/perl -s

use FindBin qw($Bin);

$::twid //= 0;
$::gray //= 0;
$::dim //= 8;

my $program;
if ($::gray) {
  if ($::twid) {
    $program = "$Bin/ppmtoltwid";
  } else {
    $program = "$Bin/ppmtol";
  }
} else {
  if ($::twid) {
    $program = "$Bin/ppmtolabtwid";
  } else {
    $program = "$Bin/ppmtolab";
  }
}

use File::Temp qw(tempdir);

select(STDOUT);
$| = 1;

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

  my $attrtxt = <$attrfp>;
  $attrtxt =~ s/\r//g;
  chomp $attrtxt;
  my @attr = split /,/, $attrtxt;
  shift(@attr) eq "$id.jpg" or die "attrs don't match $id $attrtxt";
  @attr == 40 or die "bad attrs $attrtxt";
  for (@attr) { die "bad attr [$_]" unless $_ eq 1 or $_ eq -1; }
  @attr = map { $_ > 0 ? 1 : 0 } @attr;
  my $attrdat = pack('d*', @attr);
  length($attrdat) == 40 * 8 or die "huh";

  open(my $outfp, "| djpeg -ppm |pnmcut -top 20 -bottom 197 |pnmscale -width $dim |$program") or die;
  print $outfp $jpgdata;
  close($outfp);

  print STDOUT $attrdat;

  ++$count;
  if (defined($::max) && $count >= $::max) {
    print STDERR "all done count=$count\n";
    last;
  }
}
