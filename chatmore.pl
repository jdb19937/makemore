#!/usr/bin/perl -s

use FindBin qw($Bin);
use IPC::Open2 qw(open2);

use lib "$Bin";

my ($sin, $sout);
my $pid = open2($sin, $sout, "$Bin/scriptmore");

my $module = $::m // 'Morescript';
eval "require $module";
warn $@ if $@;

our $iters;
my $interpreter = $module->new(\&ask);

sub ask {
  return "..." if ++$iters > 64;

  my $req = shift;
  print $sout "$req\n";
  warn(('  ' x $depth) . "req=$req\n");

  my $rsp = <$sin>;
  chomp($rsp);
  warn(('  ' x $depth) . "rsp=$rsp\n");

  $depth++;
  $rsp = $interpreter->interpret($rsp);
  $depth--;

  $rsp
}

select(STDOUT);
$| = 1;

while (<STDIN>) {
  chomp;

  local $iters = 0;

  my $req = $_;
  my $rsp = ask($req);

  print "$rsp\n";
}
  

close($sout);
close($sin);
waitpid($pid, 0);


