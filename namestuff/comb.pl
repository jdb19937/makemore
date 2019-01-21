chomp for my @pre = `cat presurnoms.txt`;
chomp for my @sur = `cat sursurnoms.txt`;
sub pick { $_[int rand(int @_)] }
print "const char *surnoms[] = {\n";
for (1 .. 30000 ){
  my $pre = pick(@pre);
  my $sur = pick(@sur);
  ++$surnom{"$pre$sur"};
}

for (sort keys %surnom) {
  print "  \"$_\",\n";
}

print "  NULL\n};\n";
