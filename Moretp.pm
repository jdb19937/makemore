package Moretp;


sub encode {
  my $meta = sub { /\s/ || /^</ || length($_) > 255 };
  join(' ', map { $meta->() ? '<' . length($_) : $_ } @_) .
  "\n" .
  join('', grep $meta->(), @_)
}

sub read {
  my $fp = shift;

  local $/ = "\n";
  my $line = <$fp>;
  $line ne "" or return undef;
  substr($line, -1, 1) eq "\n" or return undef;
  chop $line;

  my @word = split /\s+/, $line;
  my $bytes = 0;
  for (@word) {
    if (/^</) {
      my $bytes = int(substr($_, 1));
      if ($bytes > 0) {
        local $/ = \$bytes;
        $_ = <$fp>;
        length($_) eq $bytes or return undef;
      } else {
        $_ = '';
      }
    }
  }

  @word
}

1
