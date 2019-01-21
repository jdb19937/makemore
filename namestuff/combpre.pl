
sub pick { $_[int rand(int @_)] }

chomp for my @mpre = `cat mpreprenoms.txt`;
chomp for my @msur = `cat msurprenoms.txt`;

chomp for my @fpre = `cat fpreprenoms.txt`;
chomp for my @fsur = `cat fsurprenoms.txt`;

for (1 .. 20000 ){
  my $mpre = pick(@mpre);
  my $msur = pick(@msur);
  my $mnom = "$mpre$msur";
  $sex{$mnom} //= 1;

  my $fpre = pick(@fpre);
  my $fsur = pick(@fsur);
  my $fnom = "$fpre$fsur";
  $sex{$fnom} //= 0;
}

my $countm = 0; my $countf = 0;

print "const char *prenoms[] = {\n";
for (sort keys %sex) {
  print "  \"$_\",\n";
  if ($sex{$_}) { ++$countm } else { ++$countf };
}
print "  NULL\n};\n\n";

print STDERR "countm=$countm\ncountf=$countf\n";

print "char prenom_gender[] = {\n";
for (sort keys %sex) {
  print "  $sex{$_},\n";
}
print "  -1\n};\n";
