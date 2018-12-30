#!/usr/bin/perl -s

use FindBin qw($Bin);
use File::Temp qw(tempdir);

select(STDOUT);
$| = 1;

my $celeba = "$Bin/celeba-dataset";
my $attrfn = "$celeba/list_attr_celeba.csv";
open(my $attrfp, "< $attrfn") or die "$0: $attrfn: $!";
scalar <$attrfp>;

my $count = 0;

for my $id ('000001' .. '202599') {
  my $attrtxt = <$attrfp>;
  $attrtxt =~ s/\r//g;
  chomp $attrtxt;
  my @attr = split /,/, $attrtxt;
  shift(@attr) eq "$id.jpg" or die "attrs don't match $id $attrtxt";
  @attr == 40 or die "bad attrs $attrtxt";
  for (@attr) { die "bad attr [$_]" unless $_ eq 1 or $_ eq -1; }
  @attr = map { $_ > 0 ? 1 : 0 } @attr;

  @attr = ($attr[20]);
  my $attrdat = pack('d*', @attr);
  length($attrdat) == 8 or die "huh";
  
  print STDOUT $attrdat;

  ++$count;

  if (defined($::max) && $count >= $::max) {
    print STDERR "all done count=$count\n";
    last;
  }
}
