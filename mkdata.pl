#!/usr/bin/perl -s

use FindBin qw($Bin);
my $program = join(' ', @ARGV);

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

  open(my $outfp, "| djpeg -ppm |pnmcut -top 20 -bottom 197 |pnmscale -width 128 |./ppmtolab |$program") or die;
  print $outfp $jpgdata;
  close($outfp);

  ++$count;
  if (defined($::max) && $count >= $::max) {
    print STDERR "all done count=$count\n";
    last;
  }
}
