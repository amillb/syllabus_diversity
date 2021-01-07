# syllabus_diversity
This tool helps you analyze the diversity of assigned authors on course syllabi

It takes a list of assigned readings, and estimates the proportions of authors by gender and race/ethnicity.

For details of the methodology and results for urban sustainability courses, see:

Millard-Ball, Adam; Desai, Garima; and Fahrney, Jessica (2021). ["Diversifying Planning Education through Course Readings."](https://jpe.sagepub.com) *Journal of Planning Education and Research*, in press.

# How to install
Python 3.7 is required, along with the following Python packages. More recent versions may also work, but have not been tested. 

```
numpy 1.16.4
pandas 1.0.3
ethnicolr 0.4.0
gender_guesser 0.4.0
```

Install them with:

`pip install numpy pandas ethnicolr gender_guesser`

Note that `ethnicolr` does not currently (Dec 2020) support Python 3.8. [See here for the latest.](https://github.com/appeler/ethnicolr/issues/29)

# Preparing your input file
Prepare your file with the list of authors or assigned readings using one of the two provided templates: `template1.csv` or `template2.csv`. Both .csv and Excel (.xlsx) formats are accepted, but for Excel, only the first sheet will be used.

In both cases, authors' names must be comma-separated, e.g. `Agyeman, Julian`. Authors without a comma will be ignored as they are assumed to be an institutional author (e.g. United Nations). 

## Option 1
`template1.csv` has one row for each reading. The (optional) `courseid` column enables you to compare results for different courses. It can be numeric or the name of a course. If you just have one course, you can omit this. The `reading` column is a citation in the following format: a list of authors (last name, first name) separated by semicolons, with ** after the last author. For example:

> Agyeman, Julian; Bullard, Robert; Evans, Bob** 2003. Just Sustainabilities: Development in an Unequal World. Cambridge, MA: MIT Press.

Everything after the ** is ignored

## Option 2
`template2.csv` has one row for each author. The `readingid` column is an identifier for each reading. This allows multi-authored publications to be weighted appropriately. The (optional) `courseid` column enables you to compare different courses. If you just have one course, you can omit this. The `author` column is the author's name in (lastname, firstname) format. For example:

> Agyeman, Julian
>
> Bullard, Robert

# Running the analysis
In the command line, type:
`python analyze_readings.py inputfilename.csv [outputfilename.csv]`

Summary statistics for each course will be displayed. If you include the optional `outputfilename.csv`, a new file will be created with the estimated gender and race/ethnicity for each author or reading.

Note that the estimates for race/ethnicity are probabilistic, i.e. the probability that an author falls into a given category. This works best with large numbers of authors.

# Questions?
We would love to hear about how this tool is being used. If you have any problems, please contact [Adam Millard-Ball](https://people.ucsc.edu/~adammb/) or open a GitHub issue.