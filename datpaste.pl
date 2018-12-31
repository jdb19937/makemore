#!/usr/bin/perl

my (@fn, @fp, @k);
my $i = 0;

while (my ($fn, $k) = splice(@ARGV, 0, 2)) {
  $k > 0 or die;

  open(my $fp, "< $fn") or die "$0: $fn: $!";

  $k[$i] = $k;
  $fn[$i] = $fn;
  $fp[$i] = $fp;

  ++$i;
}

ROW: while (1) {
  for my $i (0 .. $#fp) {
    my $k = $k[$i];
    my $fp = $fp[$i];

    local $/ = \($k * 8);
    my $row = <$fp>;
    if (!defined($row)) {
      if ($i != 0) {
        die "short file $fn[$i]";
      }
      last ROW;
    }
    if (length($row) != $k * 8) {
      die "corrupt file $fn[$i]";
    }

    print $row;
  }
}


for my $i (1 .. $#fp) {
  my $k = $k[$i];
  my $fp = $fp[$i];

  local $/ = \($k * 8);
  my $row = <$fp>;
  if (defined($row)) {
    die "long file $fn[$i]";
  }
}

