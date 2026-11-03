FROM apache/hive:4.0.0
USER root
COPY postgresql-42.7.3.jar /opt/hive/lib/postgresql-jdbc.jar
COPY hadoop-aws-3.3.6.jar /opt/hive/lib/hadoop-aws.jar
COPY aws-java-sdk-bundle-1.12.367.jar /opt/hive/lib/aws-java-sdk-bundle.jar
USER hive