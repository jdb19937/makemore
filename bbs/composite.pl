#!/usr/bin/perl

chdir("/home/dan/makemore/urbs/crimea/home/bbs") or die;
system("pngtopnm base.png > base.ppm");

open(my $in, "< notes.txt") or die;

my $z = 0;
while (<$in>) {
  chomp;
  my ($cmd, $url, $x, $y) = split /\s+/;
  next unless $cmd eq 'note';

  $x = int($x);
  $y = int($y);

  $pos{$url} = [$z, $x, $y];
  ++$z;
}

my @url = sort { $a->[0] <=> $b->[0] } keys %pos;

open(my $out, "> notes.txt.tmp") or die;
my $ii = 0;
my $nn = @url;

for my $url (@url) {
  my $pos = $pos{$url};
  my ($z, $x, $y) = @$pos;

  if ($ii < $nn - 32) {
    my $inv = ($url =~ /button.png/ ? "| pnminvert" : "");
    system("curl -s '$url' |pngtopnm $inv > tmp.ppm") and die;
    system("pnmcomp -xoff=$x -yoff=$y tmp.ppm base.ppm > newbase.ppm") and die;
    system("mv -f newbase.ppm base.ppm") and die;
    warn "handled $url";
  } else {
    if ($x > -1024 && $y > -1024) {
      print $out join(" ", "note", $url, $x, $y) . "\n";
    }
  }

  ++$ii;
}
close($out);

system("convert base.ppm png:base.png.tmp") and die;
unlink("base.ppm");
unlink("newbase.ppm");
unlink("tmp.ppm");
rename("base.png.tmp", "base.png");
unlink("base.png.tmp");
rename("notes.txt.tmp", "notes.txt");
