#!/usr/bin/perl

use FindBin qw($Bin);
use lib "$Bin";
use CGI;

select(STDOUT);
$| = 1;
print "Content-type: image/jpeg\r\n\r\n";

sub show_default {
  open(my $fp, "< $Bin/default.jpg") or die;
  print while (<$fp>);
  close($fp);
  exit 1;
}

my $cgi = new CGI;
my $scale = $cgi->param('scale');
$scale = int($scale);
if ($scale < 1) {
  $scale = 1;
}
if ($scale > 16) {
  $scale = 16;
}

my ($name) = ($ENV{PATH_INFO} =~/^\/(.+$)/) or show_default();
$name =~ s/\.jpg$//;
length($name) > 0 or die;
length($name) < 32 or die;

use IO::Socket::INET;
my $s = new IO::Socket::INET 'localhost:8888' or show_default();
my $req = pack('C40', 3, (0) x 7, unpack('C*', $name));

my $n = 64 * 64 * 3 * 8;
print $s $req;

$/ = \$n;
my $lab = <$s>;
length($lab) == $n or show_default();

open(my $jpeg, "| $Bin/labtoppm 64 64 |pnmscale $scale | cjpeg -quality 95");
print $jpeg $lab;
