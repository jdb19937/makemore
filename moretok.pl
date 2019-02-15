#!/usr/bin/perl

use FindBin qw($Bin);

my %seen;
++$seen{''};
++$seen{'*'};

{
  my $branecode_fn = "$Bin/branecode.tsv";
  open(my $branecode_fp, "<", $branecode_fn) or die "$0: $branecode_fn: $!\n";
  while (<$branecode_fp>) {
    chomp;
    my ($code) = split /\t/;
    ++$seen{$code};
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
