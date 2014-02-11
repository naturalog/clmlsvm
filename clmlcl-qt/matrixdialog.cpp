#include "matrixdialog.h"
#include "ui_matrixdialog.h"
#include "defs.h"
#include <QTableWidgetItem>

MatrixDialog::MatrixDialog(QWidget *parent, const matrix& x) :
    QDialog(parent),
    ui(new Ui::MatrixDialog)
{
    ui->setupUi(this);

    ui->table->setColumnCount(x.cols());
    ui->table->setRowCount(x.rows());
    for ( int r = 0; r < x.rows(); r++ )
        for ( int c = 0; c < x.cols(); c++ )
            ui->table->setItem(r, c, new QTableWidgetItem(QString::number(x(r,c))));
}

MatrixDialog::~MatrixDialog()
{
    delete ui;
}
