#!/usr/bin/perl

use FindBin qw($Bin);

my %seen;
++$seen{''};
++$seen{'*'};

{
  my $branegasm_fn = "$Bin/branegasm.tsv";
  open(my $branegasm_fp, "<", $branegasm_fn) or die "$0: $branegasm_fn: $!\n";
  while (<$branegasm_fp>) {
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
