#!/usr/bin/perl

use strict;

# get confusion matrix
sub parse_predictions {
    my $ref_arff = shift;
    my $pred_arff = shift;
    my $pred_index = shift;
    my $lab_index = shift;
    my $cm_ref = shift;
    my @pred;
    my @ref;
    my $i = 0;
    my $data = 0;
    open(PRED, "<$pred_arff") or die "$pred_arff: $!";
    while(<PRED>) {
        chomp;
        if (/wav/){
            my @els = split(/,/);
            $pred[$i] = $els[$pred_index];
            ++$i;
        }
    }
    close(PRED);
    open(REF, "<$ref_arff") or die "$ref_arff: $!";
    my $data = 0;
    my $i2 = 0;
    while (<REF>) {
        chomp;
        if (/\@data/) {
            $data = 1;
        }
        elsif ($data && !/^\s*$/) {
            my @els = split(/,/);
            $ref[$i2] = $els[$lab_index - 1];
            ++$i2;
        }
    }
    close(REF);
    if ($i != $i2) {
        print "ERROR: Mismatched number of predictions ($i) and ground truth labels ($i2)!\n";
        exit 1;
    }
    for my $j (0..$#pred) {
        #print "ref = $ref[$j], pred = $pred[$j]\n";
        ++$cm_ref->{$ref[$j]}{$pred[$j]};
    }
}

sub score_cm
{
    my $cm_ref = shift;
    my $ignore_list = shift;
    my $ua = 0;
    my $tr = 0;
    my $n_all = 0;
    my $nclass = 0;
    my %rec;
    for my $class (keys %$cm_ref) {
        next if ($ignore_list->{$class});
        my $n = 0;
        for my $class2 (keys %{$cm_ref->{$class}}) {
            $n += $cm_ref->{$class}{$class2};
        }
        my $rec = $cm_ref->{$class}{$class} / $n;
        $rec{$class} = $rec;
        $ua += $rec;
        $tr += $cm_ref->{$class}{$class};
        $n_all += $n;
        ++$nclass;
    }
    $ua /= $nclass;
    my $wa = $tr / $n_all;
    return ($wa, $ua, \%rec);
}

my %cm;
my $pred_index = 1; # set to 2 if frame index is included

if ($#ARGV < 2) {
    print "Usage: $0 <ref_arff> <pred_arff> <lab-index> [ignore-list]\n";
    exit -1;
}

my ($ref_arff, $pred_arff, $lab_index, $ignore_list) = @ARGV;

my $ai = 0;
my $class_list = "";
open(ARFF, $ref_arff) or die "$ref_arff: $!";
while (<ARFF>) {
    if (/\@attribute\s+(\S+)\s+(\S+)/) {
        ++$ai;
        if ($ai == $lab_index) {
            $class_list = $2;
        }
    }
    if (/\@data/) { last; }
}
close(ARFF);
if (!$class_list) {
    die "Class attribute #$lab_index not found in $ref_arff!";
}
$class_list =~ s/^\s*\{\s*//;
$class_list =~ s/\s*\}\s*$//;
my @class_names = split(",", $class_list);
map { s/^\s+//; s/\s+$//; } @class_names;
#@class_names = sort(@class_names);
my $nclass = scalar @class_names;

my %ignore_list;
if ($ignore_list ne "") {
    if ($ignore_list eq "last") {
        $ignore_list{$class_names[$#class_names]} = 1;
    }
    else {
        my @ignore_list = split(",", $ignore_list);
        for (@ignore_list) {
            $ignore_list{$_} = 1;
        }
    }
}

my $max_lab_len = 0;
for (@class_names) {
    if (length($_) > $max_lab_len) {
        $max_lab_len = length($_);
    }
}
my $lab_width = 8; #$max_lab_len + 1;

parse_predictions($ref_arff, $pred_arff, $pred_index, $lab_index, \%cm);
my ($wa, $ua, $rec_ref) = score_cm(\%cm, \%ignore_list);
print "Accuracy = ", sprintf("%.1f", $wa * 100), "%\n";
print "UAR = ", sprintf("%.1f", $ua * 100), "%\n";
for my $class (@class_names) {
    next if ($ignore_list{$class});
    print "Recall ($class) = ", sprintf("%d", $rec_ref->{$class} * 100), "%\n";
}

print "Confusion matrix:\n";
print "          ", join("", map { $ignore_list{$_} ? undef : sprintf("%${lab_width}s", $_) } @class_names), "\n";
for my $class1 (@class_names) {
    next if ($ignore_list{$class1});
    printf("%${lab_width}s", $class1);
    for my $class2 (@class_names) {
        next if ($ignore_list{$class2});
        printf("%${lab_width}d", $cm{$class1}{$class2});
    }
    print "\n";
}


