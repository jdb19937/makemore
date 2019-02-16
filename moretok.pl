#!/usr/bin/perl

use FindBin qw($Bin);

my %seen;
++$seen{''};
++$seen{'*'};

{
  my $moregasm_fn = "$Bin/moregasm.tsv";
  open(my $moregasm_fp, "<", $moregasm_fn) or die "$0: $moregasm_fn: $!\n";
  while (<$moregasm_fp>) {
    chomp;
    my ($word) = split /\t/;
    ++$seen{$word};
  }
  close($branecode_fp);
}

$| = 1;

while (<>) {
  chomp;

  for (split /[\s:\(\),]+/) {
    next if /^->/;
    next if $seen{$_};
    ++$seen{$_};
    print "$_\n";
  }
}
